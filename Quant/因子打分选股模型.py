# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
from gm.api import *
from pandas import DataFrame
import statsmodels.api as sm
import pandas as pd
import talib as ta
import datetime

'''
因子打分选股模型
每一期计算股票池中股票的当期、上一期因子数据，再计算两期之间股票的收益率。
对每个因子，用上一期因子数据以及间隔期股票收益计算相关系数即IC。
以IC值作为权重，对当期每个股票，按各因子数据的加权平均数作为得分。
调仓至因子得分topN的股票，份额为最大交易金额/N。

假设1：IC值有效，即股票的截面间收益差异可以被因子的截面间差异解释，且这种关系是线性的。
假设2：所选因子是有效的。

问题1：假设1未必满足，对于假设1，可尝试运用其他指标（非相关系数），以及其他模型（非线性模型）。
问题2：假设2未必满足，可嵌套入其他模型，如因子选择模型，用以删选当期或者近期那些因子有效（更强）。
'''

def init(context):
    # 每月第一个交易日的09:40 定时执行algo任务
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
    # 数据滑窗
    context.date = 40
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
    context.factor_cut = {'BM':3,'MV':2,'EVEBITDA':5,
                          'ROEAVG':5,'SCOSTRT':5,'SGPMARGIN':5,'QUICKRT':5,'ROIC':5,'TATURNRT':5,
                          'MACD1226':5,'EMA26':5,'MA12':5,'RSI24':5
                          }
    # 所用因子溢价构造方式，大减小-0 OR 小减大-1
    context.factor_way = {'BM':0,'MV':1}
    context.period = 0

def algo(context):
    
    print(context.now)

    ############################################################################################################
    ########################## 因子数据下载、处理、计算部分，获取当期因子及收盘价数据 ###############################
    ############################################################################################################
    # 获取上一个交易日的日期
    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=last_day)
    # 获取沪深300成份股
    context.stock300 = get_history_constituents(index=context.index, start_date=last_day,
                                                end_date=last_day)[0]['constituents'].keys()
    # 获取当天有交易的股票
    not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
    # 交易天数不足股票存储器
    not_enough = []
    # 收集所有可交易标的的收益率序列 - 顺便更新not_suspended！
    ## 一次性全部读取，后续再清洗
    not_suspended_str = ''
    for i in not_suspended:
        not_suspended_str += i+','
    not_suspended_str = not_suspended_str[:-1]
    start_time = trade_day_list[-(context.date+1)]
    return_df_all = history(symbol=not_suspended_str, frequency='1d', start_time=start_time, end_time=last_day, fields='close,symbol,eob',
                          skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
    ## 计算收益率，清楚交易天数不足的
    cnt = 0
    for symbol in not_suspended:
        # 计算收益率
        return_df = return_df_all[return_df_all['symbol']==symbol]
        close = return_df.copy()
        close['date'] = close['eob'].apply(lambda x: x.strftime('%Y-%m-%d'))
        close['return'] = np.log(close['close'] / close['close'].shift(1))
        close = close.dropna()[['symbol','return','close','date']]
        if len(close) != context.date:
            not_enough.append(symbol)
            continue
        cnt += 1
        if cnt == 1:
            save_allreturn_df = close
        else:
            save_allreturn_df = pd.concat([save_allreturn_df,close],axis=0)   
    ## 保存前一天收盘价
    now_close = save_allreturn_df[save_allreturn_df['date'] == last_day][['symbol','close']]
    now_close.index = now_close['symbol']
    now_close = now_close[['close']]
    ## 更新当天交易股票的列表，从中剔除交易天数不够的
    for not_enough_stock in not_enough:
        not_suspended.remove(not_enough_stock)
    
    ############################## 因子所用指标下载，选用不同因子需要少量修改 ##############################
    # 0 #
    fin_temp_0 = get_fundamentals(table='tq_sk_finindic', symbols=not_suspended, start_date=last_day, end_date=last_day,
                           fields='PB,NEGOTIABLEMV,EVEBITDA', df=True)
    fin_temp_0['BM'] = (fin_temp_0['PB'] ** -1)
    fin_temp_0['MV'] = fin_temp_0['NEGOTIABLEMV']
    fin_temp_0.index = fin_temp_0['symbol']
    # 1 #
    fin_temp_1 = get_fundamentals(table='deriv_finance_indicator', symbols=not_suspended, start_date=last_day, end_date=last_day,
                           fields='ROEAVG,SCOSTRT,SGPMARGIN,QUICKRT,ROIC,TATURNRT', df=True)
    fin_temp_1.index = fin_temp_1['symbol']
    # 2 #
    fin_temp_2 = pd.DataFrame(index=not_suspended,columns=['MACD1226','EMA26','MA12','RSI24'],dtype=np.float)
    ta_cnt = 0
    for stock in not_suspended:
        price_series = save_allreturn_df[save_allreturn_df['symbol']==stock]['close'].values
        factor_MACD1226 = ta.MACD(price_series)[0][-1]
        factor_EMA26 = ta.EMA(price_series,26)[-1]
        factor_MA12 = ta.MA(price_series,12)[-1]
        factor_RSI24 = ta.RSI(price_series,24)[-1]
        fin_temp_2.loc[stock,:] = factor_MACD1226,factor_EMA26,factor_MA12,factor_RSI24
        ta_cnt += 1
    fin_temp_2.index = not_suspended
    #####
    # 所有因子合并
    fin = pd.concat([fin_temp_0,fin_temp_1,fin_temp_2],axis=1,sort=False)
    # 列仅留下因子，代码置入index中
    fin = fin[context.factor_list]
    # 标准化
    fin = fin.apply(lambda x: (x-np.mean(x))/np.std(x))
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    
    ############################################################################################################
    ########## 因子打分计算模块准备部分，需要用到上一期的因子以及收盘价数据，可以跨期传递，但第一期需要单独算 ###########
    ############################################################################################################
    if context.period == 0:
        # 上一期第一个交易日计算
        first_day_month = datetime.date(context.now.year, context.now.month, 1)
        last_day_premonth = first_day_month - datetime.timedelta(days = 1) #timedelta是一个不错的函数
        first_day_premonth = datetime.date(last_day_premonth.year, last_day_premonth.month, 1)
        first_day_premonth = first_day_premonth.strftime('%Y-%m-%d')
        trade_day_df = pd.DataFrame(trade_day_list,columns=['trade_day'])
        pre_last_day = trade_day_df[trade_day_df['trade_day']>=first_day_premonth]['trade_day'].values[0]
        pre_trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=pre_last_day)
        # 重复构建本期因子的步骤 last_day 替换为 pre_last_day
        context.stock300_pre = get_history_constituents(index=context.index, start_date=pre_last_day,end_date=pre_last_day)[0]['constituents'].keys()
        pre_not_suspended = get_history_instruments(symbols=context.stock300_pre, start_date=pre_last_day, end_date=pre_last_day)
        pre_not_suspended = [item['symbol'] for item in pre_not_suspended if not item['is_suspended']]
        pre_not_enough = []
        pre_not_suspended_str = ''
        for i in pre_not_suspended:
            pre_not_suspended_str += i+','
        pre_not_suspended_str = pre_not_suspended_str[:-1]
        pre_start_time = pre_trade_day_list[-(context.date+1)]
        pre_return_df_all = history(symbol=pre_not_suspended_str, frequency='1d', start_time=pre_start_time, end_time=pre_last_day, fields='close,symbol,eob',skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
        pre_cnt = 0
        for symbol in pre_not_suspended:
            pre_return_df = pre_return_df_all[pre_return_df_all['symbol']==symbol]
            close = pre_return_df.copy()                                                                                                                                    
            close['date'] = close['eob'].apply(lambda x: x.strftime('%Y-%m-%d'))
            close['return'] = np.log(close['close'] / close['close'].shift(1))
            close = close.dropna()[['symbol','return','close','date']]
            if len(close) != context.date:
                pre_not_enough.append(symbol)
                continue
            pre_cnt += 1
            if pre_cnt == 1:
                pre_save_allreturn_df = close
            else:
                pre_save_allreturn_df = pd.concat([pre_save_allreturn_df,close],axis=0)   
        pre_close = pre_save_allreturn_df[pre_save_allreturn_df['date'] == pre_last_day][['symbol','close']]
        pre_close.index = pre_close['symbol']
        pre_close = pre_close[['close']]
        for pre_not_enough_stock in pre_not_enough:
            pre_not_suspended.remove(pre_not_enough_stock)      
        ########################### 上一期因子所用指标下载，选用不同因子需要少量修改 ##############################
        # 0 #
        pre_fin_temp_0 = get_fundamentals(table='tq_sk_finindic', symbols=pre_not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                            fields='PB,NEGOTIABLEMV,EVEBITDA', df=True)
        pre_fin_temp_0['BM'] = (pre_fin_temp_0['PB'] ** -1)
        pre_fin_temp_0['MV'] = pre_fin_temp_0['NEGOTIABLEMV']
        pre_fin_temp_0.index = pre_fin_temp_0['symbol']
        # 1 #
        pre_fin_temp_1 = get_fundamentals(table='deriv_finance_indicator', symbols=pre_not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                            fields='ROEAVG,SCOSTRT,SGPMARGIN,QUICKRT,ROIC,TATURNRT', df=True)
        pre_fin_temp_1.index = pre_fin_temp_1['symbol']
        # 2 #
        pre_fin_temp_2 = pd.DataFrame(index=pre_not_suspended,columns=['MACD1226','EMA26','MA12','RSI24'],dtype=np.float)
        pre_ta_cnt = 0
        for stock in pre_not_suspended:
            price_series = pre_save_allreturn_df[pre_save_allreturn_df['symbol']==stock]['close'].values
            pre_factor_MACD1226 = ta.MACD(price_series)[0][-1]
            pre_factor_EMA26 = ta.EMA(price_series,26)[-1]
            pre_factor_MA12 = ta.MA(price_series,12)[-1]
            pre_factor_RSI24 = ta.RSI(price_series,24)[-1]
            pre_fin_temp_2.loc[stock,:] = pre_factor_MACD1226,pre_factor_EMA26,pre_factor_MA12,pre_factor_RSI24
            pre_ta_cnt += 1
        pre_fin_temp_2.index = pre_not_suspended

        pre_fin = pd.concat([pre_fin_temp_0,pre_fin_temp_1,pre_fin_temp_2],axis=1,sort=False)
        pre_fin = pre_fin[context.factor_list]
        pre_fin = pre_fin.apply(lambda x: (x-np.mean(x))/np.std(x))
        # 储存本期的因子数据以及收盘价数据，供下一期使用
        context.pre_fin, context.pre_close = fin, now_close
    else:
        # 不是第一期，则读取上一期储存的因子数据及收盘价数据copy()
        pre_fin,pre_close = context.pre_fin.copy(), context.pre_close.copy()
        # 将本期因子数据及收盘价储存供下期使用
        context.pre_fin, context.pre_close = fin, now_close
    context.period += 1
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    ##################### 因子打分模块，对每个因子计算IC值，再对每个股票计算IC加权因子得分 ##########################
    ############################################################################################################
    # 计算间隔期收益率
    close_pre_close_df = pd.merge(now_close,pre_close,left_index=True,right_index=True,how='inner')
    period_return = pd.DataFrame(close_pre_close_df['close_x']/close_pre_close_df['close_y']-1)
    period_return.index = close_pre_close_df.index
    period_return.columns = ['period_return']
    pre_fin_return = pd.merge(pre_fin,period_return,left_index=True,right_index=True,how='inner')
    # 计算因子值与间隔期收益率横截面相关系数，即IC值
    IC_list = []
    corr_matrix = pre_fin_return.corr()
    for factor in context.factor_list:
        corrcoef = corr_matrix.loc[factor,'period_return']
        IC_list.append(corrcoef)
    # 因子打分，IC加权
    IC_weight = [i/sum(IC_list) for i in IC_list]
    factor_score_df = pd.DataFrame(index=pre_fin_return.index,columns=['factor_score'])
    ## 用每个股票的当期因子数据对当期可交易股票打分
    for stock in not_suspended:
        stock_factor_values = fin.loc[stock,:].values
        score = sum(list(map(lambda x1,x2: x1*x2,stock_factor_values,IC_weight)))
        factor_score_df.loc[stock,:] = score
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    ##################################### 打分后选择分数最高的topN等量买入 #######################################
    ############################################################################################################
    sorted_factor_score_df = factor_score_df.sort_values('factor_score',ascending=False).head(context.topN)
    stock_buy = sorted_factor_score_df.index.tolist()
    positions = context.account().positions()
    ## 平不在标的池的股票
    for position in positions:
        symbol = position['symbol']
        if symbol not in stock_buy:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print('市价单平不在标的池股票', symbol)
    ## 获取股票的权重
    percent = context.ratio / len(stock_buy)
    ## 买在标的池中的股票
    for symbol in stock_buy:
        order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(symbol, '以市价单调多仓到仓位', percent)

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
    run(strategy_id='c24d0f8a-7f05-11e9-a472-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2011-01-01 08:00:00',
        backtest_end_time='2019-05-22 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)