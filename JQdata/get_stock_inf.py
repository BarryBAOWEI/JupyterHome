# -*- coding:utf-8 -*-
import tushare as ts
import pandas as pd
import datetime

################### 子函数 ###################
def stock_inf_get(L,start_date,end_date):
    """
    L         : list 股票代码的列表，例如['000002','000423',...]
    start_date: str  数据起始时间，格式YYYY-MM-DD
    end_date  : str  数据结束时间，格式YYYY-MM-DD
    """
    dict_stock_all = {}
    for stock_code in L:
        df = ts.get_k_data(stock_code,start = start_date,end = end_date)[['code','date','open','close','volume']]
        df['return'] = df['close'].pct_change()
        dict_stock_all[stock_code] = df
    return dict_stock_all

def stock_to_excel(outputpath,df_dict):
    """
    outputpath: str  输出路径，例如'c:/XXX/YYYYY'
    df_dict   : dict stock_inf_get函数得到的结果字典
    """
    outputfile = outputpath + '/stock_inf.xlsx'
    writer = pd.ExcelWriter(outputfile)
    for key, value in df_dict.items():
        value.to_excel(writer,key,index=False)
    writer.save()
################### 子函数 ###################

################### 启动函数,跑它就行 ###################
def run_all(L,start_date,end_date,outputpath):
    """
    参数说明
    L         : list 股票代码的列表，例如['000002','000423',...]
    start_date: str  数据起始时间，格式YYYY-MM-DD
    end_date  : str  数据结束时间，格式YYYY-MM-DD
    outputpath: str  输出路径，例如'c:/XXX/YYYYY'
    """
    df_dict = stock_inf_get(L,start_date,end_date)
    stock_to_excel(outputpath,df_dict)
    return df_dict
#########################################################

#['600518','000538','000963','601607','600332','000153','000650','600196','600436','600085','300267','600572','600511']
#康美药业，云南白药，华东医药，上海医药，白云山，丰原药业，仁和药业，复兴药业，片仔癀，同仁堂，尔康制药，康恩贝，国药股份

all_stock_dict = run_all(
    L = ['600518','000538','000963','601607','600332','000153','000650','600196','600436','600085','300267','600572','600511']
    ,start_date = '2015-01-01'
    ,end_date = '2017-12-31'
    ,outputpath = 'C:/Users/43460/Desktop')

# python 的 DataFrame 结果全部保存在 all_stock_dict 中，以字典的形式
# 以 df1 = all_stock_dict['600518'] 的方式就可以读取其中某一个股票的DataFrame并保存在df1中



