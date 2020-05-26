# Qirui
Excessive gaming prediction COG 2020

One can first use the code in "Web Crawler" to crawl the log data from LOL and PUBG. Please get a Userlist first which is then used to obtain corresponding data.

Code in "Pre-processing" can help us compute the observation period and pre-process the log data into time-series data. One can change the paramter of "pred" in pre-process.py to get the corresponding target week data. For example, pred = 1 will get a 'one day in advance data'. Please compute the observation period first which was set with default = 14 days.

Running addiction_ae.py to reconstruct the input, if you want to test the performances of DAE-LSTM or DAE-BI LSTM.
If one wants to directly test the performances,please use the TestData which contains 8 datasets. 7 of them, pp1-pp7, are  time-series without rebuilding. If you seek the complete dataset we used and don't want to crawl data by yourselves, please contact wuq43@mcmaster.ca.

Finally,running the code in deep learning to predict excessive gaming.

#############################
The hyperparameters we used:

-> DAE

    Network: input size * 256 -sigmoid- 256*128 -sigmoid- feature vector -sigmoid- 128*256 - sigmoid- 256 * input siz
    
    LR:e-3
    
-> RNN

    Network: input -batch normaliztion- RNN cell -dropout- RNN cell -dropout- RNN cell -batch normaliztion- 64 * 6 -SoftMax
    
    LR:4e-4
    
    Hidden Nodes:64
    
    Dropout:0.5
    
    batch_size:100
    
    time step:14
    
    input size:24
    
 -> LSTM
 
    Network: input -batch normaliztion- LSTM cell -dropout- LSTM cell -batch normaliztion- 64 * 6 -SoftMax
    LR:4e-4
    
    Hidden Nodes:64
    
    Dropout:0.5
    
    batch_size: week number
    
    time step:14
    
    input size:24
    
  -> Bi-LSTM
  
    Network: input -batch normaliztion- BiLSTM cell -dropout- BiLSTM cell -batch normaliztion- 64 * 6 -SoftMax
    LR:4e-4
    
    Hidden Nodes:64
    
    Dropout:0.5
    
    batch_size: week number
    
    time step:14
    
    input size:24
    
#############################
![image](https://github.com/LelouchWu/Qirui/survey.png)
