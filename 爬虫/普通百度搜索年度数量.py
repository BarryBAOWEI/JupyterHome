# -*- coding:utf-8 -*-
import requests, bs4, sys, webbrowser, urllib, time, re
import pandas as pd

def dateRange(st, ed):
    ststamp = str(int(time.mktime(time.strptime(st, "%Y-%m-%d"))))
    edstamp = str(int(time.mktime(time.strptime(ed, "%Y-%m-%d"))))
    return ststamp, edstamp

def yearListY(year):
    yearlst = [year+'-'+'01-01',year+'-'+'12-31']
    return yearlst

def ConsequancecountY(keywords, year):

    # all = open(txt_file, 'a')

    yearlst = yearListY(year)

    # wordss = '''“''' + keywords + '''”'''
    wordss = keywords

    dict_word_cnt = {} # 存放这一年结果的字典，年份：新闻量

    startd, endd = dateRange(yearlst[0],yearlst[1])

    # 读取百度搜索网页首页
    url = 'http://www.baidu.com.cn/s?wd=' + urllib.parse.quote(wordss) + '&rsv_enter=1&gpc=stf%3D' + startd + '%2C' +endd +'%7Cstftype%3D2&tfflag=1' #word为关键词，pn是百度用来分页的..
    # print(url)
    response = requests.get(url)
    num_unfinished = re.findall('百度为您找到相关结果约(.*)个', response.text)[0]
    # print(num_unfinished)
    num_un = num_unfinished.replace(',','')
    # print(num_un)
    num = int(num_un)

    # num_unfinished = re.findall('百度为您找到相关结果约(.*?)个', soup.text)[0].split(',')
    # l_num = len(num_unfinished)
    # num = 0
    # for numm in num_unfinished:
    #     num = num + int(numm)*pow(1000,l_num-1)
    #     l_num = l_num - 1

    dict_word_cnt[year] = num

        # 写入txt
        # numss = str(num)
        # all.write(numss + '\n')

    print('----Keyword:', keywords, 'Year:', year, 'Data Collection Finished----')

    return dict_word_cnt

def DataCollectionY(keywordlst, yearlst, filepath):

    df_tol = pd.DataFrame()

    for keywords in keywordlst:

        dict_tol = {}

        for year in yearlst:

            # 汇总每一年的该关键字的数据
            dict_tol.update(ConsequancecountY(keywords, year))

        # 将数据转化为DF
        df_tmp = pd.DataFrame.from_dict(dict_tol,orient='index').T

        # 添加关键字标签
        df_tmp['KeyWord'] = keywords

        # 加到总输出表中
        df_tol = df_tol.append(df_tmp)

    #调整输出顺序
    L_order_adjustment = ['KeyWord']
    L_order_adjustment.extend(yearlst)
    print(L_order_adjustment)
    df_tol = df_tol[L_order_adjustment]
    print(df_tol)

    df_tol.to_excel(filepath+'/DataCollection.xlsx', sheet_name='Yeah')

##############################             TEST & START            ############################################

# DataCollectionY(['金融科技','云计算','大数据','科技金融','fintech','人工智能'],['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'],'C:/Users/43460/Desktop')
DataCollectionY(['贺敬瑜'],['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018'],'C:/Users/43460/Desktop')