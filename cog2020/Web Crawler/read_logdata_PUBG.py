import json
import csv
import requests
from bs4 import BeautifulSoup
from loguru import logger
from faker import Faker
import pandas as pd
import datetime
import time


#######Open the previous URLlist
urlist = (pd.read_csv('pubg_ 13484 playersURL.csv',header=None)[1].values)[1:]

#######create CSV, only run the first time
with open("PUBG_logging.csv", "w", newline="") as f:
    csv_write = csv.writer(f)
    csv_head = ["ID","GameType", "TimeAgo", "Result", "Duration"]
    csv_write.writerow(csv_head)

headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'DNT': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36',
        }

s = requests.Session()

#######Parameters
startime = '2019-02-22T00:00:00+0000'
endtime = '2019-08-22T00:00:00+0000'
looptime = 100000


########extracting function
def extract_pubgdata(URLIST,startime,endtime,looptime):
    start = datetime.datetime.now()
    count = 0
    for url in URLIST:
        count = count + 1
        try:
            res = s.get(url, headers=headers)
        except Exception as e:
            logger.error(url)
            break

        soup = BeautifulSoup(res.text, 'lxml')

        # UserID
        user_id = soup.find('div', {'id': 'userNickname'}).get('data-user_id')
        print(user_id)
        # https://pubg.op.gg/api/users/5c3cf6341e81f900019e5ab0/matches/recent
        offset = ''

        time = endtime
        for i in range(0, 100000):
            if(time > startime):
                url = 'https://pubg.op.gg/api/users/{0}/matches/recent?after={1}'.format(user_id, offset)
                logger.info(url)

                try:
                    html = s.get(url, headers=headers).text
                    response = json.loads(html, encoding='utf-8')
                except Exception as e:
                    logger.error(e)
                    logger.error(url)
                    continue

                # print(response['matches']['summary']['matches_cnt'])
                for i in range(0, response['matches']['summary']['matches_cnt']):
                    userid = user_id
                    queue_size = response['matches']['items'][i]['queue_size']
                    started_at = response['matches']['items'][i]['started_at']
                    time_survived = response['matches']['items'][i]['participant']['stats']['combat']['time_survived']
                    offset = response['matches']['items'][i]['offset']
                    rank = response['matches']['items'][i]['participant']['stats']['rank']

                    time = started_at
                    if(started_at > startime):
                        out = open("PUBG_logging.csv", "a+", newline="")
                        csv_writer = csv.writer(out, dialect="excel")
                        row = [userid,queue_size, started_at, rank, time_survived]
                        csv_writer.writerow(row)
                        out.close()
        end = datetime.datetime.now()
        timespend = (end - start).seconds
        timeneed = ((timespend/count) * (len(URLIST) - count))/60
        print("Running Time is : %s seconds" %timespend)
        print("It will take another %s minutes to finish" %timeneed)

extract_pubgdata(urlist,startime,endtime,looptime)
