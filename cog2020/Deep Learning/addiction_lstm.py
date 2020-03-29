import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pandas import DataFrame
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.contrib import rnn
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import csv
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical#change label into one-hot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()

np.random.seed(369)
tf.set_random_seed(369)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

#df = pd.read_csv('ae_lol.csv',index_col=[0],parse_dates = True)
#df = pd.read_csv('ae11.csv',index_col=[0],parse_dates = True)
df = pd.read_csv('pp6.csv',index_col=[0],parse_dates = True)
##Parameters
pred = 6
cell_type = "lstm"#hidden layer type
lr = 1e-4#learning_rate
training_epochs = 20

batch_size = 127 - (pred-1)#shape[0]/player_num
#df = df.iloc[:300*batch_size,]
print(df.shape)

input_size = 24#save one-day data as memory
timestep_size = 14#24*14=336features
hidden_size = 100#hidden nodes
layer_num = 1
features = df.iloc[:,0:len(df.columns)-1]#colums:0-335,features
addiction_label = df.iloc[:,len(df.columns)-1:]#colums:336,label
features_num=int(features.shape[1])#336
class_num = len(df['AL1'].unique())#how many unique addiction_label in this subset

##transform addiction level to one-hot matrix
data = df['AL1']
data1 = to_categorical(data)
for i in range(0,class_num):
    df['AL%s'%(i+1)] = data1[:,i:i+1]


df.iloc[:,:336] = stdsc.fit_transform(df.iloc[:,:336])
#df.iloc[:,:336] = df.iloc[:,:336]/3600
nsplit = 1140
train = df.iloc[:nsplit*batch_size-1,]
va = df.iloc[(nsplit-400)*batch_size:nsplit*batch_size-1,]
test = df.iloc[nsplit*batch_size:,]
#hidden_size = int(len(train)/(s*342))
#print(train)


##placeholder
X_input = tf.placeholder(tf.float32, [None, features_num])
y_input = tf.placeholder(tf.float32, [None, class_num])
batch_size = tf.placeholder(tf.int32, [])
keep_prob = tf.placeholder(tf.float32, [])

X = tf.reshape(X_input, [-1, timestep_size, input_size])#transform n inputs to 24*14

##build stack LSTM with n layers
def lstm_cell(cell_type, num_nodes, keep_prob):
    assert(cell_type in ["lstm", "block_lstm"], "Wrong cell type.")
    if cell_type == "lstm":
        cell = tf.contrib.rnn.BasicLSTMCell(num_nodes)
    else:
        cell = tf.contrib.rnn.LSTMBlockCell(num_nodes)

    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)#dropout
    return cell

mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(cell_type, hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple = True)


init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)#init

outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)#rnn
h_state = state[-1][1]
##prediction
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

##OPtimizer & loss
tv = tf.trainable_variables()
regularization_cost =  0.001*tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
cross_entropy = -tf.reduce_mean(y_input * tf.math.log(y_pre)) + regularization_cost
train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_input,1))#find the most posible label in y_pre and y_ture
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.compat.v1.global_variables_initializer())#global init

#tf.summary.FileWriter("logs/", sess.graph)
##build training and testing pattern
_batch_size = 124
tacc,tace,tacc_,tace_=[],[],[],[]
for i in range(training_epochs):
    for start, end in zip( range(0, len(train), _batch_size), range(_batch_size, len(train), _batch_size)):
        batch = train[start:end]#【0，107】，【107，214】。。。
        X_batch = batch.iloc[:,:336]#n*336
        y_batch = batch.iloc[:,336:]#n*class_num
        vcost, vacc,  _ = sess.run([cross_entropy, accuracy, train_op], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 0.3, batch_size: _batch_size})

    if (i+1) % 1 == 0:
        train_acc,train_cost= 0.0,0.0
        counter = 0
        for start, end in zip( range(0, len(va), _batch_size), range(_batch_size, len(va), _batch_size)):
            batcht = va[start:end]#【0，100】，【100，200】。。。。
            X_batcht = batcht.iloc[:,0:len(va.columns)-class_num]
            y_batcht = batcht.iloc[:,len(va.columns)-class_num:]

            cost, acc = sess.run([cross_entropy, accuracy], feed_dict={X_input: X_batcht, y_input: y_batcht, keep_prob: 1.0, batch_size: _batch_size})
            train_acc += acc
            train_cost += cost
            counter += 1

        tacc.append(train_acc/counter)
        tace.append(train_cost/counter)
        #print("train acc={:.6f}; test acc={:.6f}".format(acc,test_acc/counter),i,test_cost/counter,cost)
        print("Epi: {}, validation cost={:.6f}, acc={:.6f}".format((i+1),train_cost/counter,train_acc/counter))
    #for every 10 times
    if (i+1) % 1 == 0:
        test_acc,test_cost= 0.0,0.0
        counter_ = 0
        for start, end in zip( range(0, len(test), _batch_size), range(_batch_size, len(test), _batch_size)):
            batcht = test[start:end]#【0，100】，【100，200】。。。。
            X_batcht = batcht.iloc[:,0:len(test.columns)-class_num]
            y_batcht = batcht.iloc[:,len(test.columns)-class_num:]

            _cost, _acc = sess.run([cross_entropy, accuracy], feed_dict={X_input: X_batcht, y_input: y_batcht, keep_prob: 1.0, batch_size: _batch_size})
            test_acc += _acc
            test_cost += _cost
            counter_ += 1

        tacc_.append(test_acc/counter_)
        tace_.append(test_cost/counter_)
        #print("train acc={:.6f}; test acc={:.6f}".format(acc,test_acc/counter),i,test_cost/counter,cost)
        print("Epi: {}, test cost={:.6f}, acc={:.6f}".format((i+1),test_cost/counter_,test_acc/counter_))

        print('#######################')
# np.savetxt('33slacc.txt',tacc,delimiter=',')
# np.savetxt('33slace.txt',tace,delimiter=',')
# np.savetxt('33slacc_.txt',tacc_,delimiter=',')
# np.savetxt('33slace_.txt',tace_,delimiter=',')

aqa = []
dds = []
adss = df.iloc[:,:336]
prediciton = sess.run(y_pre, feed_dict={X_input: adss,keep_prob:1.0,batch_size:len(adss)})
for i in prediciton:
    bb = i.tolist()
    aa = bb.index(max(bb))
    aqa.append(aa)

for i in addiction_label.values.tolist():
    dds.append(i[0])

# dss = 0
# for i in range(len(aqa)):
#     ds = aqa[i] - dds[i]
#     if ds == 0: dss += 1
# print(dss/len(aqa))
# print(np.unique(aqa))
# print(np.unique(dds))
#
# plt.plot(range(len(tacc)),tacc,label='acc')
# plt.plot(range(len(tacc_)),tacc_,label='acc_')
# plt.legend()
# plt.grid()
# # plt.xticks(())#不要刻度
# # plt.yticks(())#不要刻度
# plt.xlabel('Train Episodes')
# plt.ylabel('PF')
# plt.show()

labels = [0,1,2,3,4,5]


y_true = dds
y_pred = aqa

tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
#np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
#offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
#
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# # show confusion matrix
# plt.savefig('../Data/confusion_matrix.png', format='png')
plt.xticks(())#不要刻度
plt.yticks(())#不要刻度
plt.show()
