import os

# third-party library
from torch.nn import init #初始化
from weigh_init import weight_init
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()

df = pd.read_csv('ae1.csv',index_col=[0],parse_dates = True)
print(df.shape)
pred = 7

torch.manual_seed(369)    # reproducible
######### Hyper Parameters
EPOCH = 10
#### LSTM
BATCH_SIZE = 127 - (pred-1)
TIME_STEP = 14          # rnn time step / image height
INPUT_SIZE = 24       # rnn input size / image width
HIDDEN_NODES = 64
CLASS_NUM = len(df['AL1'].unique())
LAYER_NUM = 2         # Depth
DROP = 0.5
LR = 4e-4               # learning rate
######### Hyper Parameters
nsplit = 1140
train = df.iloc[:nsplit*BATCH_SIZE,]
test = df.iloc[nsplit*BATCH_SIZE:,]
print(test.shape)
x = torch.from_numpy(stdsc.fit_transform(train.iloc[:,:336])).float()
y = torch.from_numpy(train.iloc[:,336].values).float()
train_data = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

test_x = torch.from_numpy(stdsc.fit_transform(test.iloc[:,:336])).float()
test_x = test_x.view(-1,TIME_STEP,INPUT_SIZE)
test_y = test.iloc[:,336].values

class RNN(nn.Module):
    def __init__(self, batch_normalization = False, bidirectional= False):
        super(RNN,self).__init__()

        self.bn_input = nn.BatchNorm1d(14) #BN for input

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_NODES,         # rnn hidden unit
            num_layers=LAYER_NUM,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=DROP,              # stacking时只在非最后一层有效
            bidirectional=bidirectional
        )

        self.bn_hidden = nn.BatchNorm1d(HIDDEN_NODES) #BN for hidden

        if bidirectional:
            self.out = nn.Linear(2*HIDDEN_NODES, CLASS_NUM)
        else:
            self.out = nn.Linear(HIDDEN_NODES, CLASS_NUM)

        self.do_bn = batch_normalization


    def forward(self, x):
        #m = nn.Dropout(p=DROP)
        # x shape (batch, time_step, input_size)
        if self.do_bn: x = self.bn_input(x)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = r_out[:, -1, :]
        # choose r_out at the last time step
        #out = m(out)
        if self.do_bn:out = self.bn_hidden(out)
        out = self.out(out)# 用于计算loss在28个（64个，10个数位的概率）内取最后一个
        return out


rnn = RNN(batch_normalization=False,bidirectional= True)

print(rnn)
rnn.apply(weight_init)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

GS = 0
for epi in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1,TIME_STEP,INPUT_SIZE)
        output = rnn(b_x)
        b_y = b_y.long()
        # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()
        GS += 1

        if GS % 50 == 0:
            rnn.eval() #停止dropout 和 batchnormal
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            rnn.train() #开放dropout 和 batchnormal
            print('Epoch:%s'%(epi+1), '|Step:%s'%(GS),'| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
