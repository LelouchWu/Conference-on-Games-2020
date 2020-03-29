####This file accepts refined raw data and compute the duration of train(number of features)
##csv.format: userA,2019-04-01 00	2019-04-01 01	2019-04-01 02	2019-04-01 03	2019-04-01 04	2019-04-01 05
##csv.format: lelouch,0,1500,1000,3600
import time
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


def compute_ATA_duration(csv_name):
    df = pd.read_csv(csv_name)
    hours = len(df.columns)
    time = hours * 3600
    df.set_index(["UserID"], inplace=True)
    addiction = 7*3600
    addiction_duration = []
    for i in range(len(df)):#player num
        asds = (df.iloc[i]).rolling(window=168).sum()#24*7=168
        list= []
        list2 = []

        for i in range(int(hours/24-7)):#3384/24 - 7 = 134
            list.append(int(asds[167+24*i]))


###########compute duration
            if list[i] < addiction and list[i] > 0:
                list2.append(0)
            elif list[i] > addiction:
                list2.append(1)
            elif list[i] == 0:
                list2.append(-1)
        #print(list2)
        
        ssa = [i for i,x in enumerate(list2) if x ==1]
        ssa.append(10000)
        #print(ssa[0])
        if ssa[0] < 10000 and ssa[0] > 0:
            #print(list2[:ssa[0]])
            a_count = list2[:ssa[0]].count(0)
            addiction_duration.append(a_count)

    lsum = []
    for i in addiction_duration:
        if i < 21:
            lsum.append(i)
        else:
            lsum.append(21)
    mean_duration = sum(lsum)/len(lsum)
    result = [mean_duration,lsum]
    return result

result = compute_ATA_duration('testlol.csv')
print(result[0])
length=DataFrame({'ATA': result[1]})
length.plot.hist()
plt.show()
