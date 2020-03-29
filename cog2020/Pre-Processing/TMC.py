import time,datetime
import re
import os
import pandas as pd
from pyquery import PyQuery as pq
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

### time convert
def time_to_stamp(time_d):
    timeArray = time.strptime(time_d, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def create_assist_date(datestart = None,dateend = None):##generate time chain

	if datestart is None:
		datestart = '2019-01-01'
	if dateend is None:
		dateend = datetime.datetime.now().strftime('%Y-%m-%d')

	datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
	dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
	date_list = []
	date_list.append(datestart.strftime('%Y-%m-%d'))
	while datestart<dateend:
	    datestart+=datetime.timedelta(days=+1)
	    date_list.append(datestart.strftime('%Y-%m-%d'))
	return (date_list)
datelist = create_assist_date("2019-04-01","2019-08-19")


def build_timeseries(datelist):##generate time chain year-month-day
    hourslist = []
    for i in range(24):
        hourslist.append("%02d:00:00" % i)

    timelist = ["UserID"]
    for datepoint in datelist:
        for timepoint in hourslist:
            #2019-08-12 23:51:06
            time = datepoint +" "+timepoint
            time = time[:13]
            timelist.append(time)

    time_series = pd.DataFrame(columns=timelist)#construct dataframe
    time_series.set_index(["UserID"], inplace=True)#set the index with ID
    return time_series
time_series=build_timeseries(datelist)

data = pd.read_csv('LOL_test.csv')
userID = np.unique(data['ID'])
print(len(userID))
df = data[['ID','TimeAgo','Duration']]
df['Duration'] = df['Duration'].map(lambda x:re.findall(r"\d+\.?\d*",x))
df['Duration'] = df['Duration'].map(lambda x:(int(x[0])*60 +int(x[1])))
df['TimeAgo'] = df['TimeAgo'].map(lambda x:x[:13])
g1_group = df.groupby(['ID','TimeAgo'])##dictionary group

dictext = []#[{('59fd987ea0df6f00018f1d55', '2019-02-24T05'): 1162.3310000000001},
for (k,g),m in df.groupby(['ID','TimeAgo']):
    dictext.append({(k,g): sum(m["Duration"])})

for i in dictext:#####日期判断不能是2月
    for (k,g),m in i.items():
        if int(g[6]) > 3:
            if time_to_stamp(g + ":59:59") < time_to_stamp('2019-8-19 23:59:59'):
                time_series.loc[k,g] = m
time_series.fillna(0, inplace=True)

def clean_3601(dataset):####处理多余3600秒的data
    for r in range(len(dataset)):
        for c in range(len(dataset.columns)):
            if dataset.iloc[r,c] < 3600:
                dataset.iloc[r,c] = dataset.iloc[r,c]
            else :
                dataset.iloc[r,c+1] = (dataset.iloc[r,c] - 3600) + dataset.iloc[r,c+1]
                dataset.iloc[r,c] = 3600
    return dataset
clean_3601(time_series)

def dataframe_to_csv(df,csv_name):
    test=pd.DataFrame(df)
    output_path = '/Users/wayne/Desktop/ensemble_churn_prediction/zhuzhu'
    test.to_csv(output_path + '/' + csv_name)
dataframe_to_csv(time_series,"testlol.csv")
