# -*- coding:utf-8 -*-
import requests, bs4, sys, webbrowser, urllib, time, re
import pandas as pd

def unixchange(year):
    bt = year + '-01-01 00:00:00'
    et = year + '-12-31 00:00:00'
    y0, y1 = year, year
    m0, m1 = "1","12"
    d0, d1 = "1","31"
    begin_date = bt[0:10]
    end_date = et[0:10]
    bt = str(int(time.mktime(time.strptime(bt, '%Y-%m-%d %H:%M:%S'))))
    et = str(int(time.mktime(time.strptime(et, '%Y-%m-%d %H:%M:%S'))))
    return bt, et, y0, m0, d0, y1, m1, d1, begin_date, end_date

def NewscountY(keywords, year):

    bt, et, y0, m0, d0, y1, m1, d1, begin_date, end_date = unixchange(year)

    dict_word_cnt = {}

    # 读取百度新闻 高级搜索 网页首页
    url = 'https://news.baidu.com/ns?from=news&cl=2&bt=' + bt + "&y0=" + y0 + "&m0=" + m0 + "&d0=" + d0 + "&y1=" + y1 + "&m1=" + m1 + "&d1=" + d1 + "&et=" + et + "&q1=" + urllib.parse.quote(keywords) + "&submit=%E7%99%BE%E5%BA%A6%E4%B8%80%E4%B8%8B&q3=&q4=&mt=0&lm=&s=2&begin_date=" + begin_date + "&end_date=" + end_date + "&tn=newsdy&ct1=1&ct=1&rn=20&q6="
    response = requests.get(url)
    response_lst = re.findall('找到相关新闻(.*)篇', response.text)
    if response_lst == []:
       print('似乎由于网络原因未成功匹配到数据')
       num_unfinished = '-1'
    else:
       num_unfinished = response_lst[0]

    #修缮爬取结果 可能会有 找到相关新闻约xxx篇 与 找到相关新闻xxx篇 两种情况
    if num_unfinished[0] == '约':
       num_unfinished = num_unfinished[1:]

    num_un = num_unfinished.replace(',','')

    num = int(num_un)

    dict_word_cnt[year] = num

    print('----Keyword:', keywords, 'Year:', year, 'News Collection Finished----')

    return dict_word_cnt

def NewsCollectionY(keywordlst, yearlst, filepath):

    df_tol = pd.DataFrame()

    for keywords in keywordlst:

        dict_tol = {}

        for year in yearlst:

            # 汇总每一年的该关键字的数据
            dict_tol.update(NewscountY(keywords, year))

        # 将数据转化为DF
        df_tmp = pd.DataFrame.from_dict(dict_tol,orient='index').T

        # 添加关键字标签
        df_tmp['KeyWord'] = keywords

        # 加到总输出表中
        df_tol = df_tol.append(df_tmp)

    #调整输出顺序
    L_order_adjustment = ['KeyWord']
    L_order_adjustment.extend(yearlst)
    df_tol = df_tol[L_order_adjustment]
    print(df_tol)

    df_tol.to_excel(filepath+'/NewsCollection.xlsx', sheet_name='NewsCollection', index=False)

##############################             TEST & START            ############################################
# NewsCollectionY(['金融科技','云计算','大数据','科技金融','fintech','人工智能'], ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017'], 'C:/Users/43460/Desktop')
NewsCollectionY(['股票','牛市','股灾'], ['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017'], 'C:/Users/43460/Desktop')