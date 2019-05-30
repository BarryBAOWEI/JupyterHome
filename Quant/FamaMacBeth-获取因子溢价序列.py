# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
from gm.api import *
from pandas import DataFrame
import statsmodels.api as sm
import pandas as pd
import talib as ta
import datetime
import copy

'''
FamaMacBethRegression
每个月末根据因子值排序，选出首尾各1/3股票作为多空组合，其下一个月的日度收益率序列作为因子溢价序列。
'''

def init(context):
    # 每月第一个交易日的09:40 定时执行algo任务
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
    # 数据滑窗
    context.date = 60
    # 设置开仓的最大资金量
    context.ratio = 0.8
    # 设置买入得分最高股票数
    context.topN = 10
    # 股票池 - 指数
    context.index = 'SHSE.000300'
    # 所用因子名称 - 价值类因子  基本面类因子  技术面类因子
    context.factor_list = ['BM','MV','EVEBITDA',
                           'ROEAVG','SCOSTRT','SGPMARGIN','QUICKRT','ROIC','TATURNRT',
                           'MACD1226','EMA26','MA12','RSI24'
                           ]
    # 所用因子分组数量
    context.factor_cut = {'BM':3,'MV':3,'EVEBITDA':3,
                          'ROEAVG':3,'SCOSTRT':3,'SGPMARGIN':3,'QUICKRT':3,'ROIC':3,'TATURNRT':3,
                          'MACD1226':3,'EMA26':3,'MA12':3,'RSI24':3
                          }
    # # 因子多空组合构造方式,0大为多，1小为多
    # context.factor_way = {'BM':0,'MV':1,'ROIC':0}
    # 策略运行总次数计数
    context.period = 0
    # IC序列
    context.IC_series = []
    # Rank_IC序列
    context.Rank_IC_series = []
    # 多组合收益率序列
    context.Return_Long_series = []
    # 空组合收益率序列
    context.Return_Short_series = []
    # 多空组合收益率序列
    context.Return_LS_series = []
    # 序列的时间
    context.IC_date_series = []

def algo(context):
    
    print(context.now)

    # 基本数据下载 - 个股数据，因子数据
    ## 获取四个关键日期，上月末，上月第一天，上上月末，上上月末+时间窗口 ###
    ## 1上月末
    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    ## 至当期的交易日列表
    trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=last_day)
    first_day_month = datetime.date(context.now.year, context.now.month, 1)
    last_day_premonth = first_day_month - datetime.timedelta(days = 1) #timedelta是一个不错的函数
    first_day_premonth = datetime.date(last_day_premonth.year, last_day_premonth.month, 1)
    last_day_prepremonth = first_day_premonth - datetime.timedelta(days = 1)
    first_day_premonth = first_day_premonth.strftime('%Y-%m-%d')
    last_day_prepremonth = last_day_prepremonth.strftime('%Y-%m-%d')
    trade_day_df = pd.DataFrame(trade_day_list,columns=['trade_day'])
    ## 2上月第一天
    first_day = trade_day_df[trade_day_df['trade_day']>=first_day_premonth]['trade_day'].values[0]
    ## 3上上月末
    pre_last_day = get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=last_day_prepremonth)[-1]
    ## 至上一期的交易日列表
    pre_trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=pre_last_day)
    ## 4上上月末+时间窗口
    pre_start_day = pre_trade_day_list[-(context.date+1)]
    ###################################################################

    ## 获取可交易标的
    context.stock300 = get_history_constituents(index=context.index, start_date=last_day,end_date=last_day)[0]['constituents'].keys()
    context.stock300_pre = get_history_constituents(index=context.index, start_date=pre_last_day,end_date=pre_last_day)[0]['constituents'].keys()
    
    ## 下载数据
    ## 当期数据 ##########################################################################################################
    ## 当期个股收盘价获取
    not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
    not_enough = []
    not_suspended_str = ''
    for i in not_suspended:
        not_suspended_str += i+','
    not_suspended_str = not_suspended_str[:-1]
    
    ## 手动缺失值填充，多下载几日数据，用上一日数据填充 ##
    downloadfillnadatelist = get_trading_dates(exchange='SHSE', start_date=pre_start_day, end_date=last_day)
    tol_trade_len = len(downloadfillnadatelist)
    downloadfillna = pd.DataFrame(downloadfillnadatelist,columns=['trade_day'])
    ##################################################

    ## 下载所需全时段数据
    return_df_all = history(symbol=not_suspended_str, frequency='1d', start_time=pre_start_day, end_time=last_day, fields='close,symbol,eob',
                        skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
    cnt = 0
    for symbol in not_suspended:
        return_df = return_df_all[return_df_all['symbol']==symbol]
        close = return_df.copy()
        if len(close)<context.date*0.5:
            not_enough.append(symbol)
            continue            
        close['date'] = close['eob'].apply(lambda x: x.strftime('%Y-%m-%d'))
        ## 两组数据匹配，填充缺失值 ##
        fillnadf = pd.merge(downloadfillna,close,left_on='trade_day',right_on='date',how='left')
        for col in ['date','close','symbol']:
            fillnadf[col] = fillnadf[col].fillna(method='pad')
        close = fillnadf[['trade_day','close','symbol']]
        close.columns = ['date','close','symbol']
        if len(close) != tol_trade_len:
            not_enough.append(symbol)
            continue
        ############################
        close_ = close.copy()
        close_['return'] = np.log(close['close'] / close['close'].shift(1))
        close_ = close_.dropna()[['symbol','return','close','date']]
        cnt += 1
        if cnt == 1:
            save_allreturn_df = close_
        else:
            save_allreturn_df = pd.concat([save_allreturn_df,close_],axis=0)   

    ## 更新当天交易股票的列表，从中剔除交易天数不够的
    for not_enough_stock in not_enough:
        not_suspended.remove(not_enough_stock)

    ## 上期因子计算
    pre_not_suspended = not_suspended
    pre_save_allreturn_df = save_allreturn_df[(save_allreturn_df['date']>=pre_start_day) & (save_allreturn_df['date']<=pre_last_day)]
    # 0 #
    fin_temp_0 = get_fundamentals(table='tq_sk_finindic', symbols=pre_not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                           fields='PB,NEGOTIABLEMV,EVEBITDA', df=True)
    fin_temp_0 = fin_temp_0[fin_temp_0['PB'] != 0]
    fin_temp_0['BM'] = fin_temp_0['PB'].apply(lambda x: 1/x)
    fin_temp_0['MV'] = fin_temp_0['NEGOTIABLEMV']
    fin_temp_0.index = fin_temp_0['symbol']
    # 1 #
    fin_temp_1 = get_fundamentals(table='deriv_finance_indicator', symbols=pre_not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                           fields='ROEAVG,SCOSTRT,SGPMARGIN,QUICKRT,ROIC,TATURNRT', df=True)
    fin_temp_1.index = fin_temp_1['symbol']
    # 2 #
    fin_temp_2 = pd.DataFrame(index=pre_not_suspended,columns=['MACD1226','EMA26','MA12','RSI24'],dtype=np.float)
    ta_cnt = 0
    for stock in pre_not_suspended:
        # print(pre_save_allreturn_df[pre_save_allreturn_df['symbol']==stock]['close'])
        price_series = pre_save_allreturn_df[pre_save_allreturn_df['symbol']==stock]['close'].values
        factor_MACD1226 = ta.MACD(price_series)[0][-1]
        factor_EMA26 = ta.EMA(price_series,26)[-1]
        factor_MA12 = ta.MA(price_series,12)[-1]
        factor_RSI24 = ta.RSI(price_series,24)[-1]
        fin_temp_2.loc[stock,:] = factor_MACD1226,factor_EMA26,factor_MA12,factor_RSI24
        ta_cnt += 1
    fin_temp_2.index = pre_not_suspended
    #####
    # 所有因子合并
    pre_fin = pd.concat([fin_temp_0,fin_temp_1,fin_temp_2],axis=1,sort=False)
    ## 列仅留下因子，代码置入index中
    pre_fin = pre_fin[context.factor_list]
    ## 去极值
    for col in pre_fin.columns:
        pre_fin = pre_fin.sort_values(col).iloc[int(len(pre_fin)*0.02):int(len(pre_fin)*0.98),]
    ## 标准化
    pre_fin = pre_fin.dropna()
    pre_fin = pre_fin.apply(lambda x: (x-np.mean(x))/np.std(x))
    
    ## 当期与上期收盘价数据在now_close以及pre_close中；当期与上期因子数据在fin以及pre_fin中 ##
    print(pre_fin.head(5))
    print(last_day)
    print(pre_last_day)

    ## 计算分组间断点，即按因子大小划分的头部与尾部
    threshold = {}
    for factor in context.factor_list:
        threshold[factor+'_SMALL'] = pre_fin[factor].quantile(1/context.factor_cut[factor])
        threshold[factor+'_BIG'] = pre_fin[factor].quantile(1-1/context.factor_cut[factor])

    ## 划分投资组合
    stock_bin = {}
    for factor in context.factor_list:
        stock_bin[factor]={factor+'_SMALL':[],
                                factor+'_BIG':[]
                                }
        stock_bin[factor][factor+'_SMALL'] = list(set(pre_fin[pre_fin[factor]<threshold[factor+'_SMALL']].index))
        stock_bin[factor][factor+'_BIG'] = list(set(pre_fin[pre_fin[factor]>threshold[factor+'_BIG']].index))
    
    ## 计算投资组合本期收益 - 上月第1天至上月末
    now_allreturn_df = save_allreturn_df[(save_allreturn_df['date']>=first_day) & (save_allreturn_df['date']<=last_day)]
    save_trade_day_list= get_trading_dates(exchange='SHSE', start_date=first_day, end_date=last_day)
    factor_return_save_df = pd.DataFrame(index = save_trade_day_list, columns=context.factor_list)
    for factor in context.factor_list:
        cnt = 0
        for stock in stock_bin[factor][factor+'_BIG']:  
            now_allreturn_df_sub = now_allreturn_df[now_allreturn_df['symbol'] == stock]
            return_stock = now_allreturn_df_sub['return'].values
            if cnt == 0:
                return_sum_big = return_stock
            else:
                return_sum_big = list(map(lambda x1,x2:x1+x2,return_sum_big,return_stock))
            cnt += 1
        return_sum_big_ = [i/cnt for i in return_sum_big]
        
        cnt = 0
        for stock in stock_bin[factor][factor+'_SMALL']:
            now_allreturn_df_sub = now_allreturn_df[now_allreturn_df['symbol'] == stock]
            return_stock = now_allreturn_df_sub['return'].values
            if cnt == 0:
                return_sum_small = return_stock
            else:
                return_sum_small = list(map(lambda x1,x2:x1+x2,return_sum_small,return_stock))
            cnt += 1
        return_sum_small_ = [i/cnt for i in return_sum_small]

        return_series_factor = list(map(lambda x1,x2:x1-x2,return_sum_big_,return_sum_small_))
        factor_return_save_df[factor] = return_series_factor
    
    # 保存因子溢价序列
    if context.period == 0:
        context.factor_return_save_df_all = factor_return_save_df
    else:
        context.factor_return_save_df_all = pd.concat([context.factor_return_save_df_all,factor_return_save_df],axis=0)
    
    ## 保存时间
    context.IC_date_series.append(last_day)
    ## 全部计算结束
    context.period += 1

def on_backtest_finished(context, indicator):
    
    context.factor_return_save_df_all.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/FactorPreSeries.csv',index=True)


if __name__ == '__main__':
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='ec495be3-81ea-11e9-abaa-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2006-01-01 08:00:00',
        backtest_end_time='2019-05-22 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)