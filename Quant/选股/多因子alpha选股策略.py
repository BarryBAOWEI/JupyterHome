# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
from gm.api import *
from pandas import DataFrame
import statsmodels.api as sm
import pandas as pd

'''
原内容
本策略每隔1个月定时触发,根据Fama-French三因子模型对每只股票进行回归，得到其alpha值。
假设Fama-French三因子模型可以完全解释市场，则alpha为负表明市场低估该股，因此应该买入。
策略思路：
计算市场收益率、个股的账面市值比和市值,并对后两个进行了分类,
根据分类得到的组合分别计算其市值加权收益率、SMB和HML. 
对各个股票进行回归(假设无风险收益率等于0)得到alpha值.

已经改写 - by 陆树成
原策略存在BUG，结果不是回归得到，改为回归得到（回归需要时间序列，原写法根本没有时间序列），收益率没有用风险调整，alpha估计有问题，加上了风险调整。
因子溢价的构造统一采用因子头部组合的收益率减因子尾部组合（每个因子的头尾是大小还是小大可自己设定）。
三因子可以扩展为更多因子（PS：自带市场因子MKT），需要在init()函数中添加因子的各类设定，以及在因子数据下载部分添加下载代码，其余计算均不用改。

选取alpha值小于0并为最小的10只股票进入标的池
平掉不在标的池的股票并等权买入在标的池的股票
回测数据:SHSE.000300的成份股
回测时间:2017-07-01 08:00:00到2017-10-01 16:00:00
'''

def init(context):
    # 每月第一个交易日的09:40 定时执行algo任务
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
    # 数据滑窗
    context.date = 60
    # 设置开仓的最大资金量
    context.ratio = 0.8
    # 股票池 - 指数
    context.index = 'SHSE.000300'
    # 所用因子名称
    context.factor_list = ['BM','MV']
    # 所用因子分组数量
    context.factor_cut = {'BM':3,'MV':2}
    # 所用因子构造方式，大减小-0 OR 小减大-1
    context.factor_way = {'BM':0,'MV':1}

# 计算市值加权的收益率,MV为市值的分类,BM为账目市值比的分类
# def market_value_weighted(stocks, MV, BM):
#     select = stocks[(stocks.NEGOTIABLEMV == MV) & (stocks.BM == BM)]
#     market_value = select['mv'].values
#     mv_total = np.sum(market_value)
#     mv_weighted = [mv / mv_total for mv in market_value]
#     stock_return = select['return'].values
#     # 返回市值加权的收益率的和
#     return_total = []
#     for i in range(len(mv_weighted)):
#         return_total.append(mv_weighted[i] * stock_return[i])
#     return_total = np.sum(return_total)
#     return return_total

def algo(context):
    # 获取上一个交易日的日期
    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    print(context.now)
    # 获取沪深300成份股
    context.stock300 = get_history_constituents(index=context.index, start_date=last_day,
                                                end_date=last_day)[0]['constituents'].keys()
    # 获取当天有交易的股票
    not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
    # 交易天数不足股票存储器
    not_enough = []
    
    # 收集所有可交易标的的收益率序列 - 顺便更新not_suspended！
    cnt = 0
    for symbol in not_suspended:
        # 计算收益率
        return_df = history_n(symbol=symbol, frequency='1d', count=context.date + 1, end_time=last_day, fields='close,symbol',
                          skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)
        close = return_df.copy()
        close['return'] = np.log(close['close'] / close['close'].shift(1))
        close = close.dropna()[['symbol','return']]
        if len(close) != context.date:
            not_enough.append(symbol)
            continue
        cnt += 1
        if cnt == 1:
            save_allreturn_df = close
        else:
            save_allreturn_df = pd.concat([save_allreturn_df,close],axis=0)
    ## 更新当天交易股票的列表，从中剔除交易天数不够的
    for not_enough_stock in not_enough:
        not_suspended.remove(not_enough_stock)
    
    ############################## 因子所用指标下载，选用不同因子需要少量修改 ##############################
    # 获取分组所需的基本面指标
    fin = get_fundamentals(table='tq_sk_finindic', symbols=not_suspended, start_date=last_day, end_date=last_day,
                           fields='PB,NEGOTIABLEMV', df=True)
    # 计算账面市值比,为P/B的倒数
    fin['BM'] = (fin['PB'] ** -1)
    fin['MV'] = fin['NEGOTIABLEMV']
    ####################################################################################################

    # 计算分组间断点，即按因子大小划分的头部与尾部
    threshold = {}
    for factor in context.factor_list:
        threshold[factor+'_SMALL'] = fin[factor].quantile(1/context.factor_cut[factor])
        threshold[factor+'_BIG'] = fin[factor].quantile(1-1/context.factor_cut[factor])

    # 划分投资组合
    stock_bin = {}
    for factor in context.factor_list:
        stock_bin[factor]={factor+'_SMALL':[],
                             factor+'_BIG':[]
                             }
        stock_bin[factor][factor+'_SMALL'] = list(set(fin[fin[factor]<threshold[factor+'_SMALL']]['symbol']))
        stock_bin[factor][factor+'_BIG'] = list(set(fin[fin[factor]>threshold[factor+'_BIG']]['symbol']))
    
    # 计算所有投资组合市值加权收益率序列
    bin_return_df = pd.DataFrame(columns=[],index=range(context.date))
    tol_MV = fin['MV'].sum()
    for factor in context.factor_list:
        BIG_list = stock_bin[factor][factor+'_BIG']
        SMALL_list = stock_bin[factor][factor+'_SMALL']
        cnt_BIG_list = 0
        for stock_code in BIG_list:
            cnt_BIG_list += 1
            weight = fin[fin['symbol']==stock_code]['MV'].values[0]/tol_MV
            
            weight_series = [weight for i in range(context.date)]
            return_series = save_allreturn_df[save_allreturn_df['symbol']==stock_code]['return'].values
            
            weighted_return_series = list(map(lambda x1,x2:x1*x2,list(return_series),list(weight_series)))
            # 对组合中的每支股票收益率累计求和
            if cnt_BIG_list == 1:
                sum_return_series = weighted_return_series
            else:
                sum_return_series = list(map(lambda x1,x2:x1+x2,sum_return_series,weighted_return_series))
        bin_return_df[factor+'_BIG'] = sum_return_series    
        cnt_SMALL_list = 0
        for stock_code in SMALL_list:
            cnt_SMALL_list += 1
            weight = fin[fin['symbol']==stock_code]['MV'].values[0]/tol_MV
            
            weight_series = [weight for i in range(context.date)]
            return_series = save_allreturn_df[save_allreturn_df['symbol']==stock_code]['return'].values
            
            weighted_return_series = list(map(lambda x1,x2:x1*x2,list(return_series),list(weight_series)))
            print(len(weighted_return_series))
            # 对组合中的每支股票收益率累计求和
            if cnt_SMALL_list == 1:
                sum_return_series = weighted_return_series
            else:
                sum_return_series = list(map(lambda x1,x2:x1+x2,sum_return_series,weighted_return_series))
        bin_return_df[factor+'_SMALL'] = sum_return_series
        
    # 根据投资组合构造因子溢价
    factor_return = bin_return_df.copy()
    for factor in context.factor_list:
        if context.factor_way[factor] == 0:
            factor_return[factor] = factor_return[factor+'_BIG'] - factor_return[factor+'_SMALL']
        elif context.factor_way[factor] == 1:
            factor_return[factor] = - factor_return[factor+'_BIG'] + factor_return[factor+'_SMALL']
    factor_return = factor_return[context.factor_list]
    ## 市场因子溢价
    mkt_close = history_n(symbol=context.index, frequency='1d', count=context.date + 1,
                      end_time=last_day, fields='close', skip_suspended=True,
                      fill_missing='Last', adjust=ADJUST_PREV, df=True)
    mkt_return = np.log(mkt_close['close']/mkt_close['close'].shift(1))
    mkt_return = mkt_return.dropna().values
    ## 获取无风险收益率 - 一年期国债收益率
    rf_df = pd.read_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/OneYearTreasureYield.csv',index_col = 0)
    rf_df['close'] = rf_df['close'].apply(lambda x: x/255)
    rf = rf_df[rf_df.index<=last_day][-context.date:]['close'].values
    factor_return['MKT'] = list(map(lambda x1,x2:x1-x2,mkt_return,rf))

    # 至此，因子溢价构建完毕，储存在factor_return中，列名为因子名，下面开始为策略部分

    # 对每只股票进行回归获取其alpha值
    coff_pool = []
    for stock in not_suspended:
        
        ri = save_allreturn_df[save_allreturn_df['symbol']==stock]['return'].values
        ri_rf = list(map(lambda x1,x2:x1-x2,ri,rf))

        X_all = factor_return
        X_all = sm.add_constant(X_all)

        est = sm.OLS(ri_rf,X_all)
        result = est.fit()
        alpha = list(result.params)[0]

        coff_pool.append(alpha)
    stock_alpha_df = pd.DataFrame(coff_pool,index=not_suspended,columns=['alpha'])
    
    # 获取alpha最小并且小于0的10只的股票进行操作(若少于10只则全部买入)
    stock_buy = stock_alpha_df[stock_alpha_df['alpha'] < 0].sort_values(by='alpha').head(10)
    symbols_pool = stock_buy.index.tolist()
    positions = context.account().positions()
    ## 平不在标的池的股票
    for position in positions:
        symbol = position['symbol']
        if symbol not in symbols_pool:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print('市价单平不在标的池的', symbol)
    ## 获取股票的权重
    percent = context.ratio / len(symbols_pool)
    ## 买在标的池中的股票
    for symbol in symbols_pool:
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
    run(strategy_id='0bc88671-7d78-11e9-95a6-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2011-01-01 08:00:00',
        backtest_end_time='2019-05-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)