from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq
from selenium import webdriver
import time
import re
import os
import pandas as pd
from pyquery import PyQuery as pq
import numpy as np

sample_size = 12000
first_page = 2
url_lead = "https://www.op.gg/ranking/ladder/page="#basic URL
save_path = '/Users/wayne/Desktop/ensemble_churn_prediction/zhuzhu'

def parser_lol_playername(sample_size,first_page,url_leadboard):
    lis_page = []
    skip_step = int((30010-first_page)/(sample_size/100))
    for page in range(first_page,30002,skip_step):
        lis_page.append(page)

    lis_url = []
    for page_num in lis_page:
        url_leda_page = url_leadboard + "%s" % page_num#ranks URL
        lis_url.append(url_leda_page)

    player_names = []
    for url_each in lis_url:
        browser = webdriver.Chrome()#Chrome
        try:
            browser.get(url_each)
            html=browser.page_source#website source code
            uclient = uReq(url_each)
            page_html = uclient.read()
            uclient.close()
            page_soup = soup(page_html,"html.parser")#parse
            containers = page_soup.findAll("tr",{"class":"ranking-table__row"})
            for container in containers:
                player_name = container.findAll("span")
                player_names.append((player_name[0].contents[0]).replace(' ','+'))#convert the 'space' in name to '+'

            print("The current number of sampe size : %d" % len(player_names))
        except:
            print("Connection Error %s" % url_each)
            pass
        continue

        browser.close()
    return player_names

final_list = parser_lol_playername(sample_size,first_page,url_lead)
ax = np.array(final_list)
ay = ax.reshape(-1,1)
test=pd.DataFrame(data=ay)
output_path = save_path
test.to_csv(output_path + '/' + 'LOL %d playerslist.csv' % len(final_list))
