from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
from selenium import webdriver
import time
import re
import os
import pandas as pd
from pyquery import PyQuery as pq
import numpy as np
import random

players_url = []#初始list
url_test = "https://pubg.op.gg/user/Yan_CJ09"#初始搜索URL
sleeptime = 1#搜索等待时间
reuse_batch_number = 15#多少次搜索

def get_playerURL(urlp,players_url,sleeptime):
    browser = webdriver.Chrome()#启动Chrome浏览器
    browser.get(urlp)

    buttons = browser.find_elements_by_class_name("matches-item__column--btn")#计算需要点击展开的次数
    for button in buttons:
        try:
            button.click()
            time.sleep(sleeptime)
        except:
            print('Connection error %s' % urlp)
            pass
        continue

    html=browser.page_source#html编码
    page_soup = soup(html,"html.parser")#解析
    containers = page_soup.findAll("div",{"class":"player-ranking__nickname"})#找到URL的container【20】
    for container in containers:#对每一场比赛的container
        for a in container.find_all('a', href=True):#找到a开头的href内容
            players_url.append(a['href'])

    removed_list = list(set(players_url))#去掉重复项目
    return removed_list
    browser.close()


players_url = get_playerURL(url_test,players_url,sleeptime)
url_reuse = random.sample(players_url,reuse_batch_number)
for url in url_reuse:
    players_url = get_playerURL(url,players_url,sleeptime)
    print("The current size of dataset: %d" % len(players_url))


ax = np.array(players_url)
ay = ax.reshape(-1,1)
test=pd.DataFrame(data=ay)
output_path = '/Users/wayne/Desktop/ensemble_churn_prediction/zhuzhu'
test.to_csv(output_path + '/' + 'pubg_ %d playersURL.csv' % len(players_url))
