import json
import time
import csv
import requests
from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd
import datetime

### time convert
def time_to_stamp(time_d):
    timeArray = time.strptime(time_d, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


######## log ERROR
logger.add("opgg-error.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="ERROR", enqueue=True)

headers = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'DNT': '1',
    'Host': 'www.op.gg',
    'Pragma': 'no-cache',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36',
}

#######Open the previous playerlist
name_list = pd.read_csv('LOL 12000 playerslist.csv',header=None)[1].values
url_w = "https://www.op.gg/summoner/userName="
url_list = []
for name in name_list[1:]:
    url = url_w + "%s" % name
    url_list.append(url)

#######create CSV, only run the first time
with open("LOL_logging.csv", "w", newline="") as f:
    csv_write = csv.writer(f)
    csv_head = ["ID","GameType", "TimeAgo", "Result", "Duration"]
    csv_write.writerow(csv_head)

######Parameters
URList = url_list
looptime = 100000
startimest = '2019-2-19 23:40:00'
endtimest = '2019-8-19 23:40:00'


####### parsing function
def extract_lol(startime,endtime,URList,looptime):
    startime = time_to_stamp(startime)
    endtime = time_to_stamp(endtime)
    start = datetime.datetime.now()
    count = 0
    for url_p in URList:
        count = count + 1
        s = requests.Session()
        res = s.get(url_p, headers=headers)
        soup = BeautifulSoup(res.text, 'lxml')
        # 用户ID
        summoner_id = soup.find('div', {'class': 'GameListContainer'}).get('data-summoner-id')
        print(summoner_id)
        # startInfo ，timeStamp
        new_time = endtime

        # loop number
        for i in range(0, looptime):
            if (int(new_time)>startime):
                # https://www.op.gg/summoner/matches/ajax/averageAndList/startInfo=1566205695&summonerId=25441334
                try:
                    url = 'https://www.op.gg/summoner/matches/ajax/averageAndList/startInfo={0}&summonerId={1}'.format(new_time, summoner_id)
                except Exception as e:
                    logger.error(url_p)
                    logger.error(url)
                    logger.error(e)
                    break
                print(url)

                html = s.get(url, headers=headers).text
                try:
                    response = json.loads(html, encoding='utf-8')
                except Exception as e:
                    logger.error(e)
                    break
                # timestamp
                logger.info(response['lastInfo'])
                new_time = response['lastInfo']
                soup = BeautifulSoup(response['html'], 'lxml')

                UserID = summoner_id
                GameTypes = soup.find_all('div', {'class': 'GameType'})
                TimeAgos = soup.find_all('span', {'class': '_timeago'})
                GameResults = soup.find_all('div', {'class': 'GameResult'})
                GameLengths = soup.find_all('div', {'class': 'GameLength'})

                for ID, GameType, TimeAgo, GameResult, GameLength in zip(UserID, GameTypes, TimeAgos, GameResults, GameLengths):
                    # refer timestamp to save the data
                    Time = TimeAgo.get('data-datetime')
                    if(int(Time)>startime):
                        ID = UserID
                        GameType = GameType.get_text().strip()
                        TimeAgo = TimeAgo.get_text().strip()
                        GameResult = GameResult.get_text().strip()
                        GameLength = GameLength.get_text().strip()

                        out = open("LOL_logging.csv", "a+", newline="")
                        csv_writer = csv.writer(out, dialect="excel")
                        row = [ID, GameType, TimeAgo, GameResult, GameLength]
                        csv_writer.writerow(row)
                        out.close()
        end = datetime.datetime.now()
        timespend = (end - start).seconds
        timeneed = ((timespend/count) * (len(URList) - count))/60
        print("Running Time is : %s seconds" %timespend)
        print("It will take another %s minutes to finish" %timeneed)


extract_lol(startimest,endtimest,URList,looptime)
