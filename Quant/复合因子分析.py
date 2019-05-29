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
复合因子分析模型
本质是通过多个标准化后的因子按照历史移动IC作为权重进行加权，计算复合因子，是多个因子历史表现的加权平均。
得到复合因子后，按照单因子分析的方式分析。
需要先得到每个单因子的IC序列。

因子IC值与日期对应：
月末1 -> 月末2
月末2的因子IC是月末1的因子截面数据以及月末1至月末2的收益数据相关系数，
这样保证月末2的因子IC计算所用数据中没有任何未来数据。
'''

def init(context):
    # 每月第一个交易日的09:40 定时执行algo任务
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
    # 数据滑窗
    context.date = 2
    # 设置开仓的最大资金量
    context.ratio = 0.8
    # 设置买入得分最高股票数
    context.topN = 10
    # 股票池 - 指数
    context.index = 'SHSE.000300'
    # 所用多因子名称
    context.factor_list = ['BM','MV','ROIC']
    # 复合因子多空分组数
    context.factor_cut_used = 5
    # 复合因子历史IC滚动值时间窗口长度（月）
    context.factor_wind = 24
    # 策略运行总次数计数
    context.period = 0
    # 复合因子权重序列
    context.factor_weight_df = []
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

    # 获取所有单因子的IC序列，行为时间一个月，列为因子IC
    IC_df_concat_cnt = 0
    for factor in context.factor_list:
        df_temp = pd.read_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/SingleFactorICseries'+factor+'.csv',index_col=0)[['IC']]
        df_temp.columns = [factor]
        if IC_df_concat_cnt == 0:
            context.IC_series_df = df_temp
            IC_df_concat_cnt += 1
        else:
            context.IC_series_df = pd.concat([context.IC_series_df,df_temp],axis=1)

def algo(context):
    
    print(context.now)
    
    # 选择因子属性
    factor_cut_used = context.factor_cut_used
    isTaFactor = False

    # 基本数据下载 - 个股数据，因子数据
    ## 获取当期（即上一个月末）日期，以及上期的日期
    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    ## 至当期的交易日列表
    trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=last_day)
    if context.period == 0:
        first_day_month = datetime.date(context.now.year, context.now.month, 1)
        last_day_premonth = first_day_month - datetime.timedelta(days = 1) #timedelta是一个不错的函数
        first_day_premonth = datetime.date(last_day_premonth.year, last_day_premonth.month, 1)
        first_day_premonth = first_day_premonth.strftime('%Y-%m-%d')
        trade_day_df = pd.DataFrame(trade_day_list,columns=['trade_day'])
        pre_last_day = trade_day_df[trade_day_df['trade_day']>=first_day_premonth]['trade_day'].values[0]

        context.pre_last_day = last_day
    else:
        pre_last_day = context.pre_last_day
    ## 获取可交易标的
    context.stock300 = get_history_constituents(index=context.index, start_date=last_day,end_date=last_day)[0]['constituents'].keys()
    context.stock300_pre = get_history_constituents(index=context.index, start_date=pre_last_day,end_date=pre_last_day)[0]['constituents'].keys()
    ## 至上一期的交易日列表
    pre_trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=pre_last_day)
    
    ## 当期数据 ##########################################################################################################
    ## 当期个股收盘价获取
    not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
    not_enough = []
    not_suspended_str = ''
    for i in not_suspended:
        not_suspended_str += i+','
    not_suspended_str = not_suspended_str[:-1]
    start_time = trade_day_list[-(context.date+1)]
    ## 下载N天数据
    return_df_all = history(symbol=not_suspended_str, frequency='1d', start_time=start_time, end_time=last_day, fields='close,symbol,eob',
                        skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
    cnt = 0
    for symbol in not_suspended:
        return_df = return_df_all[return_df_all['symbol']==symbol]
        close = return_df.copy()
        close['date'] = close['eob'].apply(lambda x: x.strftime('%Y-%m-%d'))
        close['return'] = np.log(close['close'] / close['close'].shift(1))
        close = close.dropna()[['symbol','return','close','date']]
        if len(close) < context.date*0.5:
            not_enough.append(symbol)
            continue
        cnt += 1
        if cnt == 1:
            save_allreturn_df = close
        else:
            save_allreturn_df = pd.concat([save_allreturn_df,close],axis=0)   
    ## 保存当期收盘价
    now_close = save_allreturn_df[save_allreturn_df['date'] == last_day][['symbol','close']]
    now_close.index = now_close['symbol']
    now_close = now_close[['close']]
    ## 更新当天交易股票的列表，从中剔除交易天数不够的
    for not_enough_stock in not_enough:
        not_suspended.remove(not_enough_stock)

    ## 上期数据 ##########################################################################################################    
    ## 上期股票数据
    pre_not_suspended = get_history_instruments(symbols=context.stock300_pre, start_date=pre_last_day, end_date=pre_last_day)
    pre_not_suspended = [item['symbol'] for item in pre_not_suspended if not item['is_suspended']]
    pre_not_enough = []
    pre_not_suspended_str = ''
    for i in pre_not_suspended:
        pre_not_suspended_str += i+','
    pre_not_suspended_str = pre_not_suspended_str[:-1]
    pre_start_time = pre_trade_day_list[-(context.date+1)]
    if context.period == 0:   
        ## 下载自上期倒数N天数据
        pre_return_df_all = history(symbol=pre_not_suspended_str, frequency='1d', start_time=pre_start_time, end_time=pre_last_day, fields='close,symbol,eob',skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
        pre_cnt = 0
        for symbol in pre_not_suspended:
            pre_return_df = pre_return_df_all[pre_return_df_all['symbol']==symbol]
            close = pre_return_df.copy()                                                                                                                                    
            close['date'] = close['eob'].apply(lambda x: x.strftime('%Y-%m-%d'))
            close['return'] = np.log(close['close'] / close['close'].shift(1))
            close = close.dropna()[['symbol','return','close','date']]
            if len(close) < context.date*0.5:
                pre_not_enough.append(symbol)
                continue
            pre_cnt += 1
            if pre_cnt == 1:
                pre_save_allreturn_df = close
            else:
                pre_save_allreturn_df = pd.concat([pre_save_allreturn_df,close],axis=0)   
        ## 保存上期收盘价
        pre_close = pre_save_allreturn_df[pre_save_allreturn_df['date'] == pre_last_day][['symbol','close']]
        pre_close.index = pre_close['symbol']
        pre_close = pre_close[['close']]
        for pre_not_enough_stock in pre_not_enough:
            pre_not_suspended.remove(pre_not_enough_stock)
    else:
        pre_close, pre_save_allreturn_df ,pre_not_suspended = context.pre_close.copy(), context.pre_save_allreturn_df.copy() ,copy.copy(context.pre_not_suspended)

    ## 上期因子计算 ##
    ## 各个单因子 ##
    ## ROCI ##
    pre_fin_0 = get_fundamentals(table='deriv_finance_indicator', symbols=pre_not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                        fields='ROIC', df=True)
    pre_fin_0.index = pre_fin_0['symbol']
    ## MV ## ## BM ##
    pre_fin_1 = get_fundamentals(table='trading_derivative_indicator', symbols=pre_not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                        fields='NEGOTIABLEMV,PB', df=True)
    pre_fin_1['MV'] = pre_fin_1['NEGOTIABLEMV']
    pre_fin_1['BM'] = 1/pre_fin_1['PB']
    pre_fin_1.index = pre_fin_1['symbol']
    ## 合并
    pre_fin = pd.concat([pre_fin_0,pre_fin_1],axis=1,sort=False)
    ## 列仅留下因子，代码置入index中
    pre_fin = pre_fin[context.factor_list]
    ## 去极值
    for col in pre_fin.columns:
        pre_fin = pre_fin.sort_values(col).iloc[int(len(pre_fin)*0.02):int(len(pre_fin)*0.98),]
    ## 标准化
    pre_fin = pre_fin.apply(lambda x: (x-np.mean(x))/np.std(x))

    ## 当期与上期收盘价数据在now_close以及pre_close中；上期因子数据在pre_fin中 ##
    print(last_day)
    print(pre_last_day)

    ## 对上上期因子pre_fin计算复合因子，用以计算IC值
    ## 计算复合因子中各因子权重
    IC_series_df = context.IC_series_df[context.IC_series_df.index<=pre_last_day][-context.factor_wind:]
    IC_wind_mean = IC_series_df.mean()
    list_IC_wind_mean = IC_wind_mean.tolist()
    factor_weight_series = [i/sum(list_IC_wind_mean) for i in list_IC_wind_mean]
    compound_factor_list = []
    for stock in pre_fin.index.tolist():
        factor_series = pre_fin.reindex([stock]).values[0]
        compound_factor = sum(list(map(lambda x1,x2:x1*x2,factor_weight_series,factor_series)))
        compound_factor_list.append(compound_factor)
    pre_fin['compound_factor'] = compound_factor_list
    
    # 保存复合因子权重序列
    context.factor_weight_df.append(factor_weight_series)

    ## 因子IC值计算
    factor_used = 'compound_factor'
    ## 计算间隔期收益率
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
    factor_used = 'compound_factor'
    
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
    save_df.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/ICseries'+factor_used+'.csv',index=True)

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
    save_df_0.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/Analysis'+factor_used+'.csv',index=True)

    # 复合权重模块
    factor_weight_df = pd.DataFrame(context.factor_weight_df,columns=context.factor_list)
    factor_weight_df.index = context.IC_date_series
    factor_weight_df.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/'+factor_used+'WeightSeries.csv',index=True)


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
        backtest_start_time='2009-01-01 08:00:00',
        backtest_end_time='2019-05-22 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)