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
FamaMacBethRegression Part2
在每个月末，利用个股过去150个交易日收益率序列作为被解释变量，利用对应日期的因子溢价序列作为解释变量进行多元线性回归（含常数项），所得因子暴露系数贝塔留作后续使用。
回归所用溢价的对应因子为BM，MV，ROE，NPGRT,EMA26。
'''

def init(context):
    # 每月第一个交易日的09:40 定时执行algo任务
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
    # 数据滑窗
    context.date = 150
    # 设置开仓的最大资金量
    context.ratio = 0.8
    # 设置买入得分最高股票数
    context.topN = 10
    # 股票池 - 指数
    context.index = 'SHSE.000300'
    # 所用因子名称 - 价值类因子  基本面类因子  技术面类因子
    context.factor_list = ['BM','MV','ROEAVG','SGPMARGIN','EMA26']
    # # 因子多空组合构造方式,0大为多，1小为多
    # context.factor_way = {'BM':0,'MV':1,'ROIC':0}
    # 策略运行总次数计数
    context.period = 0
    # 记录时间序列的日期
    context.DateSeries = []
    # 读取因子溢价序列
    context.FactorPreSeries = pd.read_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/FactorPreSeries.csv',index_col=0)

def algo(context):
    
    print(context.now)

    # 基本数据下载 - 个股数据，因子数据
    ## 获取关键日期，上月末，上月第一天，上上月末，上上月末+时间窗口 ###
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
    ## 4上上月末+时间窗口+1
    pre_start_day = pre_trade_day_list[-(context.date+1)]
    ## 5上月末+时间窗口+1
    start_day = trade_day_list[-(context.date+1)]
    ###################################################################

    ## 获取可交易标的
    context.stock300 = get_history_constituents(index=context.index, start_date=last_day,end_date=last_day)[0]['constituents'].keys()
    # context.stock300_pre = get_history_constituents(index=context.index, start_date=pre_last_day,end_date=pre_last_day)[0]['constituents'].keys()
    
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
    downloadfillnadatelist = get_trading_dates(exchange='SHSE', start_date=start_day, end_date=last_day)
    tol_trade_len = len(downloadfillnadatelist)
    downloadfillna = pd.DataFrame(downloadfillnadatelist,columns=['trade_day'])
    ##################################################
    print(start_day,last_day)
    ## 下载所需全时段数据
    return_df_all = history(symbol=not_suspended_str, frequency='1d', start_time=start_day, end_time=last_day, fields='close,symbol,eob',
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
            ## 先用之前的收盘价填充，无问题
            fillnadf[col] = fillnadf[col].fillna(method='pad')
            ## 以防数据开头便是nan，再用之后数据填充，会有一个数据有问题
            fillnadf[col] = fillnadf[col].fillna(method='backfill')
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

    ## 读取适用的因子溢价序列 - 去掉一天保证与个股数据相同，个股数据还需计算收益率少去一天
    subFactorPreSeries = context.FactorPreSeries[(context.FactorPreSeries.index>=start_day) & (context.FactorPreSeries.index<=last_day)][1:]
    ## 添加常数项
    subFactorPreSeries = sm.add_constant(subFactorPreSeries)

    ## 批量回归
    stock_beta_all = []
    for stock in not_suspended:
        
        stock_df = save_allreturn_df[save_allreturn_df['symbol'] == stock].copy()
        stock_df.index = stock_df['date']
        stock_df = stock_df[['return']]
        
        est = sm.OLS(stock_df,subFactorPreSeries)
        result = est.fit()
        stock_beta_list = [result.params[factor] for factor in context.factor_list]
        stock_beta_all.append(stock_beta_list+[stock,last_day])
        
    if context.period == 0:
        context.beta_save_df = pd.DataFrame(stock_beta_all,columns=[factor+'_beta' for factor in context.factor_list]+['symbol','date'])
    else:
        beta_save_df_temp = pd.DataFrame(stock_beta_all,columns=[factor+'_beta' for factor in context.factor_list]+['symbol','date'])
        context.beta_save_df = pd.concat([context.beta_save_df,beta_save_df_temp],axis=0)
    print(context.beta_save_df.tail(5))
    
    ## 保存时间
    context.DateSeries.append(last_day)
    ## 全部计算结束
    context.period += 1

def on_backtest_finished(context, indicator):
    
    # context.beta_save_df.index = context.DateSeries
    context.beta_save_df.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/StockBetaPanelData.csv',index=False)


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
    run(strategy_id='e96646de-82a7-11e9-bf8e-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2007-01-01 08:00:00',
        backtest_end_time='2019-05-22 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)