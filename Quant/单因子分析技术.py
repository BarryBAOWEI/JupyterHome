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
单因子分析模型
非策略，属于 获取分析所用数据 的 模型。
多空组合，因子值大为多组，小为空组。
可直接获取的因子与需用价格序列计算得到的因子数据获取方法分开写。
可得到单因子IC+IR+RankIC+多空/多/空组合的收益率/净值 的 序列。
时间段自行选取，开始时间必须为'20YY-MM-01'。
所得时间序列的日期为每月最后一天。

每次获取因子时，如果仅用于分析，仅需获取一期（上月末）因子数据即可，加上间隔期收益可以计算IC，本代码获取了两期因子（上月末与上上月末），并计算了上上月末因子IC，上月末数据未传导至下一期，
本代码这类写法便于嵌入交易算法（交易中必须获取当期因子，用来预测下一期收益（打分、复合等）。
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
    # 所用因子名称
    context.factor_list = [
                           'MACD1226','EMA26','MA12','RSI24'
                           ]
    # 所用因子分组数量
    context.factor_cut = {
                          'MACD1226':5,'EMA26':5,'MA12':5,'RSI24':5
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
    
    # 选择因子属性，因子切换后，上一期计算均要改！！
    factor_used = context.factor_list[1]
    context.factor_used = factor_used
    factor_cut_used = context.factor_cut[factor_used]

    # 基本数据下载 - 个股数据，因子数据
    ## 获取上个月末日期，以及上上个月末日期
    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    ## 至当期的交易日列表
    trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=last_day)
    if context.period == 0:
        first_day_month = datetime.date(context.now.year, context.now.month, 1)
        last_day_premonth = first_day_month - datetime.timedelta(days = 1) #timedelta是一个不错的函数
        first_day_premonth = datetime.date(last_day_premonth.year, last_day_premonth.month, 1)
        last_day_prepremonth = first_day_premonth - datetime.timedelta(days = 1)
        first_day_premonth = first_day_premonth.strftime('%Y-%m-%d')
        last_day_prepremonth = last_day_prepremonth.strftime('%Y-%m-%d')
        trade_day_df = pd.DataFrame(trade_day_list,columns=['trade_day'])
        pre_last_day = get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=last_day_prepremonth)[-1]

        context.pre_last_day = last_day
    else:
        pre_last_day = copy.copy(context.pre_last_day)

    ## 获取可交易标的
    context.stock300 = get_history_constituents(index=context.index, start_date=last_day,end_date=last_day)[0]['constituents'].keys()
    context.stock300_pre = get_history_constituents(index=context.index, start_date=pre_last_day,end_date=pre_last_day)[0]['constituents'].keys()

    ## 至上一期的交易日列表
    pre_trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=pre_last_day)

    ## 下载数据，分为下载 N 天与下载 1 天
    ## 当期数据 ##########################################################################################################
    ## 当期个股收盘价获取
    not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
    not_enough = []
    not_suspended_str = ''
    for i in not_suspended:
        not_suspended_str += i+','
    not_suspended_str = not_suspended_str[:-1]
    start_day = trade_day_list[-(context.date+1)]

    ## 手动缺失值填充，多下载几日数据，用上一日数据填充 ##
    downloadfillnadatelist = get_trading_dates(exchange='SHSE', start_date=start_day, end_date=last_day)
    tol_trade_len = len(downloadfillnadatelist)
    downloadfillna = pd.DataFrame(downloadfillnadatelist,columns=['trade_day'])
    ##################################################

    ## 下载所需全时段数据
    return_df_all = history(symbol=not_suspended_str, frequency='1d', start_time=start_day, end_time=last_day, fields='close,symbol,eob',
                        skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
    cnt = 0
    for symbol in not_suspended:
        return_df = return_df_all[return_df_all['symbol']==symbol]
        close = return_df.copy()
        if len(close) < context.date*0.5:
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
        cnt += 1
        if cnt == 1:
            save_allreturn_df = close
        else:
            save_allreturn_df = pd.concat([save_allreturn_df,close],axis=0)   
    ## 保存上个月末收盘价
    now_close = save_allreturn_df[save_allreturn_df['date'] == last_day][['symbol','close']]
    now_close.index = now_close['symbol']
    now_close = now_close[['close']]
    ## 更新当天交易股票的列表，从中剔除交易天数不够的
    for not_enough_stock in not_enough:
        not_suspended.remove(not_enough_stock)

    ## 上期数据 ##########################################################################################################    
    ## 上期股票数据
    if context.period == 0:
        pre_not_suspended = get_history_instruments(symbols=context.stock300_pre, start_date=pre_last_day, end_date=pre_last_day)
        pre_not_suspended = [item['symbol'] for item in pre_not_suspended if not item['is_suspended']]
        print(len(pre_not_suspended))
        pre_not_enough = []
        pre_not_suspended_str = ''
        for i in pre_not_suspended:
            pre_not_suspended_str += i+','
        pre_not_suspended_str = pre_not_suspended_str[:-1]
        pre_start_day = pre_trade_day_list[-(context.date+1)]
        ## 手动缺失值填充，多下载几日数据，用上一日数据填充 ##
        downloadfillnadatelist = get_trading_dates(exchange='SHSE', start_date=pre_start_day, end_date=pre_last_day)
        tol_trade_len = len(downloadfillnadatelist)
        downloadfillna = pd.DataFrame(downloadfillnadatelist,columns=['trade_day'])
        ## 下载自上上个月末倒数N天数据
        pre_return_df_all = history(symbol=pre_not_suspended_str, frequency='1d', start_time=pre_start_day, end_time=pre_last_day, fields='close,symbol,eob',skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
        pre_cnt = 0
        for symbol in pre_not_suspended:
            return_df = pre_return_df_all[pre_return_df_all['symbol']==symbol]
            close = return_df.copy()
            if len(close) < context.date*0.5:
                pre_not_enough.append(symbol)
                continue
            close['date'] = close['eob'].apply(lambda x: x.strftime('%Y-%m-%d'))
            ## 两组数据匹配，填充缺失值 ##
            fillnadf = pd.merge(downloadfillna,close,left_on='trade_day',right_on='date',how='left')
            for col in ['date','close','symbol']:
                fillnadf[col] = fillnadf[col].fillna(method='pad')
            close = fillnadf[['trade_day','close','symbol']]
            close.columns = ['date','close','symbol']
            if len(close) != tol_trade_len:
                pre_not_enough.append(symbol)
                continue
            ############################
            pre_cnt += 1
            if pre_cnt == 1:
                pre_save_allreturn_df = close
            else:
                pre_save_allreturn_df = pd.concat([pre_save_allreturn_df,close],axis=0)   
        ## 保存上上个月末收盘价
        pre_close = pre_save_allreturn_df[pre_save_allreturn_df['date'] == pre_last_day][['symbol','close']]
        pre_close.index = pre_close['symbol']
        pre_close = pre_close[['close']]
        for pre_not_enough_stock in pre_not_enough:
            pre_not_suspended.remove(pre_not_enough_stock)
    else:
        pre_close, pre_save_allreturn_df, pre_not_suspended = context.pre_close.copy(), context.pre_save_allreturn_df.copy(), copy.copy(context.pre_not_suspended)
    print(len(pre_not_suspended))
    ## 上期因子计算
    ## MACD1226 ##
    if factor_used == 'MACD1226':
        pre_fin = pd.DataFrame(index=pre_not_suspended,columns=[factor_used],dtype=np.float)
        pre_ta_cnt = 0
        for stock in pre_not_suspended:
            price_series = pre_save_allreturn_df[pre_save_allreturn_df['symbol']==stock]['close'].values
            pre_factor_te = ta.MACD(price_series)[0][-1]
            pre_fin.loc[stock,:] = pre_factor_te
            pre_ta_cnt += 1
    ## EMA26 ##
    if factor_used == 'EMA26':
        pre_fin = pd.DataFrame(index=pre_not_suspended,columns=[factor_used],dtype=np.float)
        pre_ta_cnt = 0
        for stock in pre_not_suspended:
            price_series = pre_save_allreturn_df[pre_save_allreturn_df['symbol']==stock]['close'].values
            pre_factor_te = ta.EMA(price_series,26)[-1]
            pre_fin.loc[stock,:] = pre_factor_te
            pre_ta_cnt += 1
    ## MA12 ##
    if factor_used == 'MA12':
        pre_fin = pd.DataFrame(index=pre_not_suspended,columns=[factor_used],dtype=np.float)
        pre_ta_cnt = 0
        for stock in pre_not_suspended:
            price_series = pre_save_allreturn_df[pre_save_allreturn_df['symbol']==stock]['close'].values
            pre_factor_te = ta.MA(price_series,12)[-1]
            pre_fin.loc[stock,:] = pre_factor_te
            pre_ta_cnt += 1
    ## RSI24 ##
    if factor_used == 'RSI24':
        pre_fin = pd.DataFrame(index=pre_not_suspended,columns=[factor_used],dtype=np.float)
        pre_ta_cnt = 0
        for stock in pre_not_suspended:
            price_series = pre_save_allreturn_df[pre_save_allreturn_df['symbol']==stock]['close'].values
            pre_factor_te = ta.RSI(price_series,24)[-1]
            pre_fin.loc[stock,:] = pre_factor_te
            pre_ta_cnt += 1
    ## 列仅留下因子，代码置入index中
    pre_fin = pre_fin[[factor_used]]
    ## 去极值
    for col in pre_fin.columns:
        pre_fin = pre_fin.sort_values(col).iloc[int(len(pre_fin)*0.02):int(len(pre_fin)*0.98),]    
    ## 标准化
    pre_fin = pre_fin.apply(lambda x: (x-np.mean(x))/np.std(x))

    ## 当期与上期收盘价数据在now_close以及pre_close中；当期与上期因子数据在fin以及pre_fin中 ##
    print(last_day)
    print(pre_last_day)

    ## 因子IC值计算
    ## 计算间隔期收益率
    print(len(now_close))
    print(len(pre_close))
    close_pre_close_df = pd.merge(now_close,pre_close,left_index=True,right_index=True,how='inner')
    period_return = pd.DataFrame(close_pre_close_df['close_x']/close_pre_close_df['close_y']-1)
    period_return.index = close_pre_close_df.index
    period_return.columns = ['period_return']
    pre_fin_return = pd.merge(pre_fin,period_return,left_index=True,right_index=True,how='inner')
    ## 计算因子值与间隔期收益率横截面相关系数，即IC值
    corr_matrix = pre_fin_return.corr()
    corrcoef = corr_matrix.loc[factor_used,'period_return']
    context.IC_series.append(corrcoef)
    
    ## 因子Rank_IC值计算
    pre_fin_return_s0 = pre_fin_return.sort_values(factor_used)
    pre_fin_return_s0[factor_used] = range(1,len(pre_fin_return_s0)+1)
    pre_fin_return_s1 = pre_fin_return_s0.sort_values('period_return')
    pre_fin_return_s1['period_return'] = range(1,len(pre_fin_return_s1)+1)
    ## 计算因子排名值与间隔期收益率排名横截面相关系数，即IR值
    corr_matrix = pre_fin_return_s1.corr()
    corrcoef = corr_matrix.loc[factor_used,'period_return']
    context.Rank_IC_series.append(corrcoef)

    ## 多空组合收益率计算
    ## 计算多空分组间断点，即按因子大小划分的头部与尾部
    threshold = {}
    threshold[factor_used+'_SMALL'] = pre_fin[factor_used].quantile(1/factor_cut_used)
    threshold[factor_used+'_BIG'] = pre_fin[factor_used].quantile(1-1/factor_cut_used)
    ## 划分多空投资组合，记录收盘价共下期计算收益使用
    stock_bin = {}
    stock_bin[factor_used]={factor_used+'_SMALL':[],
                            factor_used+'_BIG':[]
                            }
    stock_bin[factor_used][factor_used+'_SMALL'] = list(set(pre_fin[pre_fin[factor_used]<threshold[factor_used+'_SMALL']].index))
    stock_bin[factor_used][factor_used+'_BIG'] = list(set(pre_fin[pre_fin[factor_used]>threshold[factor_used+'_BIG']].index))
    
    ## 计算组合自上上月末到上月末的收益率
    ## 先获取 上上月末 多空组合中标的在 上上月末 以及 上月末 两期的收盘价，用以计算一个月的收益率
    pre_small_close = pre_close.reindex(stock_bin[factor_used][factor_used+'_SMALL'])
    pre_big_close = pre_close.reindex(stock_bin[factor_used][factor_used+'_BIG'])
    now_small_close = now_close.reindex(pre_small_close.index.tolist())
    now_big_close = now_close.reindex(pre_big_close.index.tolist())

    small_close = pd.merge(now_small_close,pre_small_close,left_index=True,right_index=True,how='inner')
    big_close = pd.merge(now_big_close,pre_big_close,left_index=True,right_index=True,how='inner')

    small_return = pd.DataFrame(small_close['close_x']/small_close['close_y']-1,columns=['small_return'],index=small_close.index)
    big_return = pd.DataFrame(big_close['close_x']/big_close['close_y']-1,columns=['big_return'],index=big_close.index)
    ## 组合收益是组合内标的平均收益
    small_return_r = small_return.sum().values[0]/small_return.count().values[0]
    big_return_r = big_return.sum().values[0]/big_return.count().values[0]
    
    context.Return_Short_series.append(small_return_r)
    context.Return_Long_series.append(big_return_r)
    context.Return_LS_series.append(big_return_r-small_return_r)
    
    ## 保存时间
    context.IC_date_series.append(last_day)
    # 保存当期收盘价、收盘价序列以及未停牌列表供下期使用
    context.pre_close, context.pre_save_allreturn_df ,context.pre_not_suspended = now_close.copy(), save_allreturn_df.copy(), copy.copy(not_suspended)
    ## 全部计算结束
    context.pre_last_day = last_day
    context.period += 1

def on_backtest_finished(context, indicator):
    factor_used = context.factor_used
    
    # 计算净值序列
    Return_Long_series_ = [i+1 for i in context.Return_Long_series]
    Price_Long_series = []
    for i in range(len(Return_Long_series_)):
        num = 1
        for j in Return_Long_series_[:i+1]:
            num *= j
        Price_Long_series.append(num)
    Return_Short_series_ = [i+1 for i in context.Return_Short_series]
    Price_Short_series = []
    for i in range(len(Return_Short_series_)):
        num = 1
        for j in Return_Short_series_[:i+1]:
            num *= j
        Price_Short_series.append(num)
    Return_LS_series_ = [i+1 for i in context.Return_LS_series]
    Price_LS_series = []
    for i in range(len(Return_LS_series_)):
        num = 1
        for j in Return_LS_series_[:i+1]:
            num *= j
        Price_LS_series.append(num)

    save_dict = {'IC':context.IC_series,
                 'Rank_IC':context.Rank_IC_series,
                 'Long':context.Return_Long_series,
                 'Short':context.Return_Short_series,
                 'LS':context.Return_LS_series,
                 'LongP':Price_Long_series,
                 'ShortP':Price_Short_series,
                 'LSP':Price_LS_series
                 }
    save_df = pd.DataFrame(save_dict)
    save_df.index = context.IC_date_series
    save_df['RollingIC-mean'] = save_df['IC'].rolling(5).mean()
    save_df['RollingIC-std'] = save_df['IC'].rolling(5).std()
    save_df['IR'] = save_df['RollingIC-mean']/save_df['RollingIC-std']
    save_df.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/SingleFactorICseries'+factor_used+'.csv',index=True)

    # 评价模块
    IC_mean = save_df['IC'].mean()
    IC_std = save_df['IC'].std()
    IR_mean = save_df['IR'].mean()
    IR_std = save_df['IR'].std()
    Rank_IC_mean = save_df['Rank_IC'].mean()
    Rank_IC_std = save_df['Rank_IC'].std()
    IC_isPositive = sum([1 if i>0 else 0 for i in save_df['IC'].tolist()])/len(save_df['IC'])
    IC_bigger005 = sum([1 if abs(i)>0.05 else 0 for i in save_df['IC'].tolist()])/len(save_df['IC'])
    IC_Posbigger005 = sum([1 if abs(i)>0.05 and i >0 else 0 for i in save_df['IC'].tolist()])/sum([1 if i >0 else 0 for i in save_df['IC'].tolist()])
    IC_Negbigger005 = sum([1 if abs(i)>0.05 and i <0 else 0 for i in save_df['IC'].tolist()])/sum([1 if i <0 else 0 for i in save_df['IC'].tolist()])
    Long_mean = save_df['Long'].mean()
    Long_std = save_df['Long'].std()
    Short_mean = save_df['Short'].mean()
    Short_std = save_df['Short'].std()
    LS_mean = save_df['LS'].mean()
    LS_std = save_df['LS'].std()
    ## 评价模块输出
    save_df_0 = pd.DataFrame(index=['IC_mean',
                                    'IC_std',
                                    'IR_mean',
                                    'IR_std',
                                    'Rank_IC_mean',
                                    'Rank_IC_std',
                                    'IC_isPositive',
                                    'IC_bigger005',
                                    'IC_Posbigger005',
                                    'IC_Negbigger005',
                                    'Long_mean',
                                    'Long_std',
                                    'Short_mean',
                                    'Short_std',
                                    'LS_mean',
                                    'LS_std'
                                    ],
                            columns=[factor_used]
                            )
    save_df_0[factor_used] = [IC_mean,IC_std,IR_mean,IR_std,Rank_IC_mean,Rank_IC_std,IC_isPositive,IC_bigger005,IC_Posbigger005,IC_Negbigger005,Long_mean,Long_std,Short_mean,Short_std,LS_mean,LS_std]
    save_df_0.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/SingleFactorAnalysis'+factor_used+'.csv',index=True)


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
    run(strategy_id='2e616d1e-812f-11e9-ad4e-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2017-11-01 08:00:00',
        backtest_end_time='2019-05-22 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)