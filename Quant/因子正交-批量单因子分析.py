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
因子正交
所选择的某一因子为被正交因子。
比较原因子与正交后因子表现。
正交因子选用：。
正交因子之间首先进行逐步正交。
被正交因子对全部正交因子回归得到截面残差序列作为被正交因子的正交后因子。

批量单因子写法加入。
数据下载模块重写。
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
    # 序列的时间
    context.IC_date_series = []
    # 策略运行总次数计数
    context.period = 0
    '''
    二选一：批量单因子
    '''
    # # 所用因子名称（被正交因子）
    # context.factor_list = ['EVEBITDA','ROEAVG','SCOSTRT','SGPMARGIN','QUICKRT','ROIC','NPGRT','TATURNRT','CASHRT','CURRENTRT',
    #                         'MACD1226','EMA26','MA12','RSI24']
    # # 所用因子分组数量
    # context.factor_cut = {'EVEBITDA':5,'ROEAVG':5,'SCOSTRT':5,'SGPMARGIN':5,'QUICKRT':5,'ROIC':5,'NPGRT':5,'TATURNRT':5,'CASHRT':5,'CURRENTRT':5,
    #                         'MACD1226':5,'EMA26':5,'MA12':5,'RSI24':5}
    '''
    二选一：单因子
    '''
    # 所用因子名称（被正交因子）
    context.factor_list = []
    # 所用因子分组数量
    context.factor_cut = {:}
    '''
    正交因子
    '''
    # 正交因子 - 估值、市值（规模）、增速、杠杆
    context.orthogonal_factor = ['BM','MV','TAGRT','EM']
    '''
    各类模式开关
    '''
    # 因子正交处理开关
    context.orthogonal = False
    # 批量单因子开关
    # context.manySingle = True
    '''
    原始因子评价值储存
    '''    
    # IC序列
    context.IC_series_dict = {}
    # Rank_IC序列
    context.Rank_IC_series_dict = {}
    # 多组合收益率序列
    context.Return_Long_series_dict = {}
    # 空组合收益率序列
    context.Return_Short_series_dict = {}
    # 多空组合收益率序列
    context.Return_LS_series_dict = {}
    '''
    正交处理因子评价值储存
    '''    
    # IC序列
    context.IC_series_or_dict = {}
    # Rank_IC序列
    context.Rank_IC_series_or_dict = {}
    # 多组合收益率序列
    context.Return_Long_series_or_dict = {}
    # 空组合收益率序列
    context.Return_Short_series_or_dict = {}
    # 多空组合收益率序列
    context.Return_LS_series_or_dict = {}

def algo(context):
    
    print(context.now)
    '''
    获取关键日期，上月末，上月第一天，上上月末，上上月末+时间窗口
    '''
    ## 1 上月末
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
    ## 2 上月第一天
    first_day = trade_day_df[trade_day_df['trade_day']>=first_day_premonth]['trade_day'].values[0]
    ## 3 上上月末
    pre_last_day = get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=last_day_prepremonth)[-1]
    ## 至上一期的交易日列表
    pre_trade_day_list= get_trading_dates(exchange='SHSE', start_date='2005-01-01', end_date=pre_last_day)
    ## 4 上上月末+时间窗口+1
    pre_start_day = pre_trade_day_list[-(context.date+1)]
    ## 5 上月末+时间窗口+1
    start_day = trade_day_list[-(context.date+1)]
    ###################################################################
    '''
    获取本次样本的全交易日
    '''
    thisTradeDayDf = pd.read_csv('D:/JUEJINStockData/trade_day.csv')
    thisTradeDayDf = thisTradeDayDf[(thisTradeDayDf['trade_day'] >= pre_start_day) & (thisTradeDayDf['trade_day'] <= last_day)]
    thisTradeDayList = thisTradeDayDf['trade_day'].tolist()
    '''
    获取所需全时段数据 - 上上月末+时间窗口 ~ 上月末
    '''
    ## 获取可交易标的
    ## 上上月末时的可交易标的 - 用于计算因子以及因子分组
    stock300_pre = get_history_constituents(index=context.index, start_date=pre_last_day,end_date=pre_last_day)[0]['constituents'].keys()

    ## 获取交易数据
    ## 可交易股票（上上月末)
    not_suspended = get_history_instruments(symbols=stock300_pre, start_date=pre_last_day, end_date=pre_last_day)
    not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]

    get_stock_cnt = 0
    for stock in not_suspended:
        stock_inf = pd.read_csv('D:/JUEJINStockDataV1/'+ stock +'.csv',index_col=0)
        stock_inf = stock_inf[(stock_inf.index>=pre_start_day) & (stock_inf.index<=last_day)]

        ## 有股价天数不足（上市不久），剔除
        if len(stock_inf) != len(thisTradeDayList):
            continue
        
        stock_inf['return'] = stock_inf['close'] / stock_inf['close'].shift(1) - 1
        stock_inf = stock_inf.dropna()

        if get_stock_cnt == 0:
            save_allreturn_df = stock_inf
        else:
            save_allreturn_df = pd.concat([save_allreturn_df,stock_inf],axis=0)
        get_stock_cnt += 1
    
    ## 更新可交易股票列表
    not_suspended = list(set(save_allreturn_df['symbol'].tolist()))

    ## 计算两时段之间的收益率 - 月度收益率，用以计算因子IC值等
    ## 保存上个月末收盘价
    now_close = save_allreturn_df[save_allreturn_df.index == last_day][['symbol','close']]
    now_close.index = now_close['symbol']
    now_close = now_close[['close']]  
    ## 保存上上个月末收盘价
    pre_close = save_allreturn_df[save_allreturn_df.index == pre_last_day][['symbol','close']]
    pre_close.index = pre_close['symbol']
    pre_close = pre_close[['close']]
    ## 合并计算收益率
    close_pre_close_df = pd.merge(now_close,pre_close,left_index=True,right_index=True,how='inner')
    period_return = pd.DataFrame(close_pre_close_df['close_x']/close_pre_close_df['close_y']-1)
    period_return.index = close_pre_close_df.index
    period_return.columns = ['period_return']
    '''
    二选一：因子数据获取以及因子计算 - 批量单因子
    '''
    factor_compute_df = save_allreturn_df[save_allreturn_df.index<=pre_last_day]
    # 0 #
    fin_temp_0 = get_fundamentals(table='tq_sk_finindic', symbols=not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                           fields='PB,NEGOTIABLEMV,EVEBITDA', df=True)
    fin_temp_0 = fin_temp_0[fin_temp_0['PB'] != 0]
    fin_temp_0['BM'] = fin_temp_0['PB'].apply(lambda x: 1/x)
    fin_temp_0['MV'] = np.log(fin_temp_0['NEGOTIABLEMV'])
    fin_temp_0.index = fin_temp_0['symbol']
    # 1 #
    fin_temp_1 = get_fundamentals(table='deriv_finance_indicator', symbols=not_suspended, start_date=pre_last_day, end_date=pre_last_day,
                           fields='ROEAVG,SCOSTRT,SGPMARGIN,QUICKRT,ROIC,NPGRT,TATURNRT,EM,CASHRT,CURRENTRT,TAGRT', df=True)
    fin_temp_1.index = fin_temp_1['symbol']
    # 2 #
    fin_temp_2 = pd.DataFrame(index=not_suspended,columns=['MACD1226','EMA26','MA12','RSI24'],dtype=np.float)
    ta_cnt = 0
    for stock in not_suspended:
        price_series = factor_compute_df[factor_compute_df['symbol']==stock]['close'].values
        factor_MACD1226 = ta.MACD(price_series)[0][-1]
        factor_EMA26 = ta.EMA(price_series,26)[-1]
        factor_MA12 = ta.MA(price_series,12)[-1]
        factor_RSI24 = ta.RSI(price_series,24)[-1]
        fin_temp_2.loc[stock,:] = factor_MACD1226,factor_EMA26,factor_MA12,factor_RSI24
        ta_cnt += 1
    fin_temp_2.index = not_suspended
    #####
    ## 所有因子合并
    pre_fin = pd.concat([fin_temp_0,fin_temp_1,fin_temp_2],axis=1,sort=False)
    ## 列仅留下因子，代码置入index中 - 被正交与正交因子均要保存
    pre_fin = pre_fin[context.factor_list+context.orthogonal_factor]
    ## 去极值
    for col in pre_fin.columns:
        pre_fin = pre_fin.sort_values(col).iloc[int(len(pre_fin)*0.02):int(len(pre_fin)*0.98),]
    ## 标准化
    pre_fin = pre_fin.dropna()
    pre_fin = pre_fin.apply(lambda x: (x-np.mean(x))/np.std(x))
    '''
    二选一：因子数据获取以及因子计算 - 单因子
    '''
    ## 单因子计算
    # pass
    # ## 列仅留下因子，代码置入index中 - 被正交与正交因子均要保存
    # pre_fin = pre_fin[context.factor_list+context.orthogonal_factor]
    # ## 去极值
    # for col in pre_fin.columns:
    #     pre_fin = pre_fin.sort_values(col).iloc[int(len(pre_fin)*0.02):int(len(pre_fin)*0.98),]
    # ## 标准化
    # pre_fin = pre_fin.dropna()
    # pre_fin = pre_fin.apply(lambda x: (x-np.mean(x))/np.std(x))
    
    if context.orthogonal:
        '''
        所有研究的单因子进行正交
        '''
        ## 正交因子首先进行内部正交
        orthogonal_factor_df = pre_fin[context.orthogonal_factor].copy()
        for orthogonal_cnt in range(1,len(context.orthogonal_factor)):
            orthogonal_list = context.orthogonal_factor[:orthogonal_cnt+1]
            y_name = orthogonal_list[-1]
            X_name = orthogonal_list[:-1]
            X = orthogonal_factor_df[X_name]
            y = orthogonal_factor_df[y_name]
            ## 回归获取残差序列
            X = sm.add_constant(X)
            est = sm.OLS(y,X)
            result = est.fit()
            residual = result.resid
            ## 替换为正交处理后的因子，供逐步回归
            orthogonal_factor_df[y_name] = residual

        ## 所有研究的单因子分别正交处理
        orthogonal_pre_fin = pre_fin.copy()
        for factor in context.factor_list:
            y = pre_fin[factor].tolist()
            X = orthogonal_factor_df
            ## 回归获取残差序列
            X = sm.add_constant(X)
            est = sm.OLS(y,X)
            result = est.fit()
            residual = result.resid
            ## 替换为正交处理后的因子，供逐步回归
            orthogonal_pre_fin[factor] = residual
    '''
    原始因子IC值、Rank_IC值、多空组合收益率计算
    '''
    pre_fin_return = pd.merge(pre_fin,period_return,left_index=True,right_index=True,how='inner')
    ## IC值、Rank_IC值计算
    for factor in context.factor_list:    
        ## IC
        sub_pre_fin_return = pre_fin_return[[factor,'period_return']].copy()
        ## 计算因子值与间隔期收益率横截面相关系数，即IC值
        corr_matrix = sub_pre_fin_return.corr()
        corrcoef = corr_matrix.loc[factor,'period_return']
        ## 储存该因子IC值
        if factor not in context.IC_series_dict.keys():
            context.IC_series_dict[factor] = []
        context.IC_series_dict[factor].append(corrcoef)
        
        ## Rank_IC
        for col in sub_pre_fin_return.columns.tolist():
            sub_pre_fin_return = sub_pre_fin_return.sort_values(col)
            sub_pre_fin_return[col] = range(1,len(sub_pre_fin_return)+1)
        ## 计算因子大小排名与间隔期收益率排名横截面相关系数，即Rank_IC值
        corr_matrix = sub_pre_fin_return.corr()
        corrcoef = corr_matrix.loc[factor,'period_return']
        ## 储存该因子Rank_IC值
        if factor not in context.Rank_IC_series_dict.keys():
            context.Rank_IC_series_dict[factor] = []
        context.Rank_IC_series_dict[factor].append(corrcoef)
        
        ## 多空组合收益率
        ## 计算多空分组间断点，即按因子大小划分的头部与尾部
        threshold = {}
        factor_cut_used = context.factor_cut[factor]
        threshold[factor+'_SMALL'] = pre_fin[factor].quantile(1/factor_cut_used)
        threshold[factor+'_BIG'] = pre_fin[factor].quantile(1-1/factor_cut_used)
        ## 划分多空投资组合，记录收盘价共下期计算收益使用
        stock_bin = {}
        stock_bin[factor]={factor+'_SMALL':[],
                                factor+'_BIG':[]
                                }
        stock_bin[factor][factor+'_SMALL'] = list(set(pre_fin[pre_fin[factor]<threshold[factor+'_SMALL']].index))
        stock_bin[factor][factor+'_BIG'] = list(set(pre_fin[pre_fin[factor]>threshold[factor+'_BIG']].index))       
        ## 计算组合自上上月末到上月末的收益率
        small_return = period_return.reindex(stock_bin[factor][factor+'_SMALL'])
        big_return = period_return.reindex(stock_bin[factor][factor+'_BIG'])
        ## 组合收益是组合内标的平均收益
        small_return_r = small_return.sum().values[0]/small_return.count().values[0]
        big_return_r = big_return.sum().values[0]/big_return.count().values[0]
        if factor not in context.Return_Short_series_dict.keys():
            context.Return_Short_series_dict[factor] = []
        if factor not in context.Return_Long_series_dict.keys():
            context.Return_Long_series_dict[factor] = []
        if factor not in context.Return_LS_series_dict.keys():
            context.Return_LS_series_dict[factor] = []
        ## 储存该因子多空组合收益率
        context.Return_Short_series_dict[factor].append(small_return_r)
        context.Return_Long_series_dict[factor].append(big_return_r)
        context.Return_LS_series_dict[factor].append(big_return_r-small_return_r)
    
    if context.orthogonal:
        '''
        正交处理因子IC值、Rank_IC值、多空组合收益率计算
        '''
        orthogonal_pre_fin_return = pd.merge(orthogonal_pre_fin,period_return,left_index=True,right_index=True,how='inner')
        ## IC值、Rank_IC值计算
        for factor in context.factor_list:    
            ## IC
            sub_orthogonal_pre_fin_return = orthogonal_pre_fin_return[[factor,'period_return']].copy()
            ## 计算因子值与间隔期收益率横截面相关系数，即IC值
            corr_matrix = sub_orthogonal_pre_fin_return.corr()
            corrcoef = corr_matrix.loc[factor,'period_return']
            ## 储存该因子IC值
            if factor not in context.IC_series_or_dict.keys():
                context.IC_series_or_dict[factor] = []
            context.IC_series_or_dict[factor].append(corrcoef)
            
            ## Rank_IC
            for col in sub_orthogonal_pre_fin_return.columns.tolist():
                sub_orthogonal_pre_fin_return = sub_orthogonal_pre_fin_return.sort_values(col)
                sub_orthogonal_pre_fin_return[col] = range(1,len(sub_orthogonal_pre_fin_return)+1)
            ## 计算因子大小排名与间隔期收益率排名横截面相关系数，即Rank_IC值
            corr_matrix = sub_orthogonal_pre_fin_return.corr()
            corrcoef = corr_matrix.loc[factor,'period_return']
            ## 储存该因子Rank_IC值
            if factor not in context.Rank_IC_series_or_dict.keys():
                context.Rank_IC_series_or_dict[factor] = []
            context.Rank_IC_series_or_dict[factor].append(corrcoef)
            
            ## 多空组合收益率
            ## 计算多空分组间断点，即按因子大小划分的头部与尾部
            threshold = {}
            factor_cut_used = context.factor_cut[factor]
            threshold[factor+'_SMALL'] = orthogonal_pre_fin[factor].quantile(1/factor_cut_used)
            threshold[factor+'_BIG'] = orthogonal_pre_fin[factor].quantile(1-1/factor_cut_used)
            ## 划分多空投资组合，记录收盘价共下期计算收益使用
            stock_bin = {}
            stock_bin[factor]={factor+'_SMALL':[],
                                    factor+'_BIG':[]
                                    }
            stock_bin[factor][factor+'_SMALL'] = list(set(orthogonal_pre_fin[orthogonal_pre_fin[factor]<threshold[factor+'_SMALL']].index))
            stock_bin[factor][factor+'_BIG'] = list(set(orthogonal_pre_fin[orthogonal_pre_fin[factor]>threshold[factor+'_BIG']].index))       
            ## 计算组合自上上月末到上月末的收益率
            small_return = period_return.reindex(stock_bin[factor][factor+'_SMALL'])
            big_return = period_return.reindex(stock_bin[factor][factor+'_BIG'])
            ## 组合收益是组合内标的平均收益
            small_return_r = small_return.sum().values[0]/small_return.count().values[0]
            big_return_r = big_return.sum().values[0]/big_return.count().values[0]
            if factor not in context.Return_Short_series_or_dict.keys():
                context.Return_Short_series_or_dict[factor] = []
            if factor not in context.Return_Long_series_or_dict.keys():
                context.Return_Long_series_or_dict[factor] = []
            if factor not in context.Return_LS_series_or_dict.keys():
                context.Return_LS_series_or_dict[factor] = []
            ## 储存该因子多空组合收益率
            context.Return_Short_series_or_dict[factor].append(small_return_r)
            context.Return_Long_series_or_dict[factor].append(big_return_r)
            context.Return_LS_series_or_dict[factor].append(big_return_r-small_return_r)
    
    ## 保存时间
    context.IC_date_series.append(last_day)
    ## 全部计算结束
    context.period += 1

def on_backtest_finished(context, indicator):
    '''
    原始因子结果输出
    '''
    ## 分析模块表格框架
    save_df_analysis = pd.DataFrame(index=[
                                    'IC_mean',
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
                                    ])
      
    for factor in context.factor_list:
        ## 序列数据模块输出
        save_dict = {'IC':context.IC_series_dict[factor],
                    'Rank_IC':context.Rank_IC_series_dict[factor],
                    'Long':context.Return_Long_series_dict[factor],
                    'Short':context.Return_Short_series_dict[factor],
                    'LS':context.Return_LS_series_dict[factor]
                    }
        save_df = pd.DataFrame(save_dict)
        save_df.index = context.IC_date_series
        save_df['RollingIC-mean'] = save_df['IC'].rolling(10).mean()
        save_df['RollingIC-std'] = save_df['IC'].rolling(10).std()
        save_df['IR'] = save_df['RollingIC-mean']/save_df['RollingIC-std']
        save_df.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/ICseries'+factor+'.csv',index=True)
        
        ## 分析模块输出
        IC_mean = save_df['IC'].mean()
        IC_std = save_df['IC'].std()
        IR_mean = save_df['IR'].mean()
        IR_std = save_df['IR'].std()
        Rank_IC_mean = save_df['Rank_IC'].mean()
        Rank_IC_std = save_df['Rank_IC'].std()
        IC_isPositive = sum([1 if i>0 else 0 for i in save_df['IC'].tolist()])/len(save_df['IC'])
        IC_bigger005 = sum([1 if abs(i)>0.05 else 0 for i in save_df['IC'].tolist()])/len(save_df['IC'])
        if sum([1 if i >0 else 0 for i in save_df['IC'].tolist()]) == 0:
            IC_Posbigger005 = np.nan
        else:
            IC_Posbigger005 = sum([1 if abs(i)>0.05 and i >0 else 0 for i in save_df['IC'].tolist()])/sum([1 if i >0 else 0 for i in save_df['IC'].tolist()])
        if sum([1 if i <0 else 0 for i in save_df['IC'].tolist()]) == 0:
            IC_Negbigger005 = np.nan
        else:
            IC_Negbigger005 = sum([1 if abs(i)>0.05 and i <0 else 0 for i in save_df['IC'].tolist()])/sum([1 if i <0 else 0 for i in save_df['IC'].tolist()])
        Long_mean = save_df['Long'].mean()
        Long_std = save_df['Long'].std()
        Short_mean = save_df['Short'].mean()
        Short_std = save_df['Short'].std()
        LS_mean = save_df['LS'].mean()
        LS_std = save_df['LS'].std()

        save_df_analysis[factor] = [IC_mean,IC_std,IR_mean,IR_std,Rank_IC_mean,Rank_IC_std,IC_isPositive,IC_bigger005,IC_Posbigger005,IC_Negbigger005,Long_mean,Long_std,Short_mean,Short_std,LS_mean,LS_std]
    save_df_analysis.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/ManySingleFactorsAnalysis.csv',index=True)
    
    if context.orthogonal:
        '''
        正交因子结果输出
        '''
        ## 分析模块表格框架
        save_df_analysis = pd.DataFrame(index=[
                                        'IC_mean',
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
                                        ])
        
        for factor in context.factor_list:
            ## 序列数据模块输出
            save_dict = {'IC':context.IC_series_or_dict[factor],
                        'Rank_IC':context.Rank_IC_series_or_dict[factor],
                        'Long':context.Return_Long_series_or_dict[factor],
                        'Short':context.Return_Short_series_or_dict[factor],
                        'LS':context.Return_LS_series_or_dict[factor]
                        }
            save_df = pd.DataFrame(save_dict)
            save_df.index = context.IC_date_series
            save_df['RollingIC-mean'] = save_df['IC'].rolling(10).mean()
            save_df['RollingIC-std'] = save_df['IC'].rolling(10).std()
            save_df['IR'] = save_df['RollingIC-mean']/save_df['RollingIC-std']
            save_df.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/OrICseries'+factor+'.csv',index=True)
            
            ## 分析模块输出
            IC_mean = save_df['IC'].mean()
            IC_std = save_df['IC'].std()
            IR_mean = save_df['IR'].mean()
            IR_std = save_df['IR'].std()
            Rank_IC_mean = save_df['Rank_IC'].mean()
            Rank_IC_std = save_df['Rank_IC'].std()
            IC_isPositive = sum([1 if i>0 else 0 for i in save_df['IC'].tolist()])/len(save_df['IC'])
            IC_bigger005 = sum([1 if abs(i)>0.05 else 0 for i in save_df['IC'].tolist()])/len(save_df['IC'])
            if sum([1 if i >0 else 0 for i in save_df['IC'].tolist()]) == 0:
                IC_Posbigger005 = np.nan
            else:
                IC_Posbigger005 = sum([1 if abs(i)>0.05 and i >0 else 0 for i in save_df['IC'].tolist()])/sum([1 if i >0 else 0 for i in save_df['IC'].tolist()])
            if sum([1 if i <0 else 0 for i in save_df['IC'].tolist()]) == 0:
                IC_Negbigger005 = np.nan
            else:
                IC_Negbigger005 = sum([1 if abs(i)>0.05 and i <0 else 0 for i in save_df['IC'].tolist()])/sum([1 if i <0 else 0 for i in save_df['IC'].tolist()])
            Long_mean = save_df['Long'].mean()
            Long_std = save_df['Long'].std()
            Short_mean = save_df['Short'].mean()
            Short_std = save_df['Short'].std()
            LS_mean = save_df['LS'].mean()
            LS_std = save_df['LS'].std()

            save_df_analysis[factor] = [IC_mean,IC_std,IR_mean,IR_std,Rank_IC_mean,Rank_IC_std,IC_isPositive,IC_bigger005,IC_Posbigger005,IC_Negbigger005,Long_mean,Long_std,Short_mean,Short_std,LS_mean,LS_std]
        save_df_analysis.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/ManyOrSingleFactorsAnalysis.csv',index=True)

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
        backtest_start_time='2006-01-01 08:00:00',
        backtest_end_time='2008-04-22 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)