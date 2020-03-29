####This file accepts refined raw data and train duration, then transform it into final data with addiciton scale(label)
##input:csv.format: userA,2019-04-01 00	2019-04-01 01	2019-04-01 02	2019-04-01 03	2019-04-01 04	2019-04-01 05
##csv.format: lelouch,0,1500,1000,3600
##output format:index,1:00:00	1:01:00	1:02:00	1:03:00	1:04:00	1:05:00	1:06:00	1:07:00.....addiction_level
##output format:0,1500,1000,3600,.....,1

import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


features = []
hours = ["%02d:00" % i for i in range(24)]#00-23
days = ["%02d:" % i for i in range(1,15)]#1-14
for day in days:
    for hour in hours:
        features.append(day + hour)
features.append('Addiction Level')
new_df = pd.DataFrame(columns=features)#construct dataframe


df = pd.read_csv('testlol.csv')
print(df.shape)
hours = len(df.columns)
time = hours * 3600
df.set_index(["UserID"], inplace=True)
#df['totaltime'] = df.sum(axis=1)
addiction = 7*3600
addiction2 = 14*3600
addiction3 = 20*3600
addiction4 = 30*3600
addiction5 = 40*3600
addiction_duration = 2*7
index = 0
#count = 0
for player in range(len(df)):#player num,for each player
    rw = (df.iloc[player]).rolling(window=168).sum()#24*7=168，compute seven-day time_d,每隔1小时取一周数据
    list= []
    list2 = []
    list3 = []
    list4 = []
    for w in range(int(hours/24-7)):#3384/24 - 7 = 134,for each window，总共有多少天再减去初始7天
        list.append(int(rw[167+24*w]))#rolling in rolling,从第一个7天开始，每隔1天再取7天，将每隔1天取一周数据拉成list

        if list[w] > addiction5:#addiction playing bound 5，4，3，2，1
           list2.append(5)
        elif list[w] > addiction4:
            list2.append(4)
        elif list[w] > addiction3:
            list2.append(3)
        elif list[w] > addiction2:
            list2.append(2)
        elif list[w] > addiction:
            list2.append(1)
        elif list[w] > 0:#normal playing bound
            list2.append(0)
        else:
            list2.append(0)#no data

    #print(list2)#APB:5,4,3,2,1  NPB:0, None:-1

    addiction_index = [(i2,x2) for i2,x2 in enumerate(list2) if x2!=-1]#extract index of all data besides -1,0
    addiction_index.append((-1,-1))#in case of list = [none]
    #print(addiction_index)#返回不是-1的所有data的index

    for i3,x3 in addiction_index:
        if i3 > (addiction_duration-1):#delete the data whose duratoin < addiction_duration
            if sum(list2[i3-addiction_duration:i3-1]) != -addiction_duration:#delete the null data
                list3.append((i3,x3))
    #print(list3)#[(58, 0), (59, 0), 第几周的数据和label

    for i4,x4 in list3:
        (daf ,dab) = (i4-addiction_duration)*24, i4*24
        list4.append((daf ,dab,x4))


    #print(len(list4))#[(1056, 1391, 0), (1080, 1415, 0),....需要取数据的区间和label
    #count = count +len(list4)

    for i5 in list4:
        ssa = df.iloc[player,int(i5[0]):int(i5[1])]
        ssa['Addiction Level']=i5[2]
        new_df.loc[index] = ssa.values
        index = index+1

print(new_df.info())
# def dataframe_to_csv(df,csv_name):
#     test=pd.DataFrame(df)
#     output_path = '/Users/wayne/Desktop/ensemble_churn_prediction/zhuzhu'
#     test.to_csv(output_path + '/' + csv_name)
# dataframe_to_csv(new_df,"pre-process_lol.csv")
