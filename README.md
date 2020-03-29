# Qirui
Video Game Addiction For COG 2020

One can first use the code in "Web Crawler" to crawl the log data from LOL and PUBG. Please get a Userlist first which is then used to obtain corresponding data.

Code in "Pre-processing" can help us compute the observation period and pre-process the log data into time-series data. One can change the paramter of "pred" in pre-process.py to get the corresponding target week data. For example, pred = 1 will get a 'one day in advance data'. Please compute the observation period first which was set with default = 14 days.

Running addiction_ae.py to reconstruct the input, if you want to test the performances of DAE-LSTM or DAE-Stacked LSTM.
If one wants to directly test the performances,please use the TestData which contains 8 datasets. 7 of them, pp1-pp7, are  time-series without rebuilding. 

Finally,running addiction_lstm.py to predict video game addiciton. One can change the parameter of layer_num to build stacked LSTM with different depth.
