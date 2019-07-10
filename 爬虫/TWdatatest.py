import pandas as pd
import requests
import os
import datetime
import numpy as np
import time

day_list = pd.date_range('2007-07-01','2019-07-01')

URL = 'https://www.twse.com.tw/fund/BFI82U'

HEADER = {'Accept': 'application/json, text/javascript, */*; q=0.01',
'Accept-Encoding': 'gzip, deflate, br',
'Accept-Language': 'zh-CN,zh;q=0.9',
'Connection': 'keep-alive',
'Cookie': '_ga=GA1.3.1163449717.1560416439; JSESSIONID=6A9A12283C537A8C29DABD4362EBC101; _gid=GA1.3.1612463192.1562040349; _gat=1',
'Host': 'www.twse.com.tw',
'Referer': 'https://www.twse.com.tw/zh/page/trading/fund/BFI82U.html',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.81 Safari/537.36',
'X-Requested-With': 'XMLHttpRequest'}

num_cnt = 0
day_cnt = 0
### first time shut down ###
save_df = pd.read_csv('C:/Users/jxjsj/Desktop/twdata.csv')

for day in day_list:
    str_day = str(day)[:4]+str(day)[5:7]+str(day)[8:10]
    stamp = str(round(time.time() * 1000))
    
    ### first time shut down ###
    if str_day in [str(i) for i in save_df['day'].unique().tolist()]:
        day_cnt += 1
        num_cnt = len(save_df)
        continue
    
    query = {
                'response': 'json',
                'dayDate': str_day,
                'weekDate': '20190701',
                'monthDate': '20190701',
                'type': 'day',
                '_': stamp
                                }
    reloading = 0
    while True:
        reloading += 1
        if reloading > 5:
            break
        try:
            r = requests.post(URL, query, HEADER)
            break
        except Exception as e:
            print(e)
            continue
    try:
        if day_cnt == 0:
            table_title = r.json()['fields']
            save_df = pd.DataFrame(columns=['day']+table_title)
        data = r.json()['data']
        r.close()
        data_col_num = len(data)

        for i in range(data_col_num):
            data_i = data[i]
            save_df.loc[num_cnt,:] = [str_day]+data_i
            num_cnt += 1
            print(str_day,'Complete!')
            time.sleep(1)

    except:
        continue
    day_cnt += 1
    if day_cnt%10 == 0:
        save_df.sort_values('day').to_csv('C:/Users/jxjsj/Desktop/twdata.csv',index=False)