#####This file accpets labeled data and uses autoencoder to reconstruct 336features
import pandas as pd
import time
from pandas import DataFrame
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
import csv
import os
import datetime
from sklearn.model_selection import train_test_split#sklearn里的，分离函数，将一个dataset分成train和test
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()

np.random.seed(369)
tf.set_random_seed(369)
output_path = '/Users/wayne/Desktop/ensemble_churn_prediction/zhuzhu'
csv_name = "ae7.csv"
df = pd.read_csv('pp.csv',index_col=[0],parse_dates = True)
pred = 7
length = 127 - (pred-1)#shape[0]/player_num
df = df.iloc[:300*length,]
features = df.iloc[:,0:len(df.columns)-1]#colums:0-335,features
features = stdsc.fit_transform(features)
addiction_label = df.iloc[:,len(df.columns)-1:]#colums:336,label

######### Hyper Parameters
learning_rate = 0.001#Adam training——rate
training_epochs = 500#number of epochs
batch_size = 121#size of the dataset per time of one-epoch
display_step = 10#print the index of cost which divide display_step
######### Hyper Parameters
train, test = df,df

x_train = train.iloc[:,0:len(df.columns)-1]
x_train = stdsc.fit_transform(x_train)
y_train = train.iloc[:,len(df.columns)-1:]

x_test = test.iloc[:,0:len(df.columns)-1]
x_test  = stdsc.fit_transform(x_test )
y_test = test.iloc[:,len(df.columns)-1:]

######### Hyper Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 =  128# 2nd layer num features
######### Hyper Parameters
n_input = x_train.shape[1] # input size
#print(x_train.describe())

sess = tf.InteractiveSession()
#staring to build autoencoder
X = tf.placeholder("float", [None, n_input])#input placeholder

#weight dict
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
#biases dict
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    # Encoder first layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder second layer with sigmoid activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Decoder first layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    # Decoder second layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


#AE Parameters
encoder_op = encoder(X)#input->code layer
decoder_op = decoder(encoder_op)#code layer->new_input
y_pred = decoder_op
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))#loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)#RMSRrop

# Initializing the variables
init = tf.global_variables_initializer()
sess.run(init)

#build learning structure
loss = []
for epoch in range(training_epochs):
    for start, end in zip( range(0, len(features), batch_size), range(batch_size, len(features), batch_size)):#这里要注意end会早一步结束
        batch_xs = features[start:end]#【0，100】，【100，200】。。。。
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    loss.append(c)
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch),"cost=", "{:.9f}".format(c))
print("Optimization Finished!")

np.savetxt('ss.txt',loss,delimiter=',')

##rebuild dataframe with ae data
def build_recon_df(input,output,y_pred):
    start = datetime.datetime.now()
    features = []
    hours = ["%02d:00" % i for i in range(24)]#00-23
    days = ["%02d:" % i for i in range(1,15)]#1-14
    for day in days:
        for hour in hours:
            features.append(day + hour)
    features.append('AL1')
    new_df = pd.DataFrame(columns=features)#construct dataframe

    encode_decode = sess.run(y_pred, feed_dict={X: input})
    for i in range(0,len(input)):
        ssa = np.append(encode_decode[i],(output.iloc[i]).values)
        new_df.loc[i] = ssa
        if i == 0: new_df.to_csv(output_path + '/' + csv_name,mode='a')
        else: new_df.to_csv(output_path + '/' + csv_name,mode='a', header=False)
        new_df = pd.DataFrame(columns=features)#清零减少内存
        if i % 500 == 0:
            end = datetime.datetime.now()
            timespend = (end - start).seconds
            timeneed = int(((timespend/(i+1)) * (184283 - i))/60)
            print("Running Time is : %s seconds" %timespend)
            print("It will take another %s minutes to finish" %timeneed)
            print('%s条数据'%(i))
    return new_df

tf.summary.FileWriter("logs/", sess.graph)
build_recon_df(features,addiction_label,y_pred)


plt.plot(range(len(loss)),loss,label='Sigmoid,Sigmoid')
plt.legend()
plt.grid()
# plt.xticks(())#不要刻度
# plt.yticks(())#不要刻度
plt.xlabel('Train Episodes')
plt.ylabel('Average Loss')
plt.show()

# sr = np.loadtxt("sigmoid-relu.txt")
# tr = np.loadtxt("tanh-relu.txt")
# tt = np.loadtxt("tanh2.txt")
#
# #plt.style.use('dark_background')
#plt.plot(range(len(s2)),s2,label='Sigmoid,Sigmoid')
# plt.plot(range(len(s2)),sr,label='Relu,Sigmoid')
# plt.plot(range(len(s2)),tr,label='Relu,Tanh')
# plt.plot(range(len(s2)),tt,label='Tanh,Tanh')
# plt.legend()
# plt.grid()
# # plt.xticks(())#不要刻度
# # plt.yticks(())#不要刻度
# plt.xlabel('Train Episodes')
# plt.ylabel('Average Loss')
# plt.show()
