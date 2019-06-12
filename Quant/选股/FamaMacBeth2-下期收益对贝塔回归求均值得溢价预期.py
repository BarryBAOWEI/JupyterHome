# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import datetime
import pandas as pd
import statsmodels.api as sm

'''
FamaMacBethRegression Part3
在每个时间点，将个股下个月度收益率与本月末贝塔系数回归，得到因子溢价，再计算月均溢价及其T值。
'''

def init(context):
    
    factor_list = ['BM','MV','ROEAVG','SGPMARGIN','EMA26'] 
    factor_beta_list = [factor+'_beta' for factor in factor_list]
    
    # 读取beta面板数据并获取回归时点
    BetaPanel = pd.read_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/StockBetaPanelData.csv')[['BM_beta','MV_beta','ROEAVG_beta','SGPMARGIN_beta','EMA26_beta','symbol','date']]
    date_df = pd.DataFrame(list(set(BetaPanel['date'].tolist())),columns=['date'])
    date_df = date_df.sort_values('date')
    date_list = date_df['date'].tolist()
    
    # 对每个时点，做截面回归 - 贝塔的最后一个月未的数据用上，因为回归的是下一月的未收集数据
    factor_premiumn_df = pd.DataFrame(index = date_list[:-1], columns = factor_beta_list)
    for date_cnt in range(len(date_list)-1):
        
        date = date_list[date_cnt]
        date_next = date_list[date_cnt+1]
        
        # 获取beta数据
        BetaCross = BetaPanel[BetaPanel['date'] == date].copy()
        BetaCross.index = BetaCross['symbol']
        BetaCross = BetaCross[['BM_beta','MV_beta','ROEAVG_beta','SGPMARGIN_beta','EMA26_beta']]
        stock_list = BetaCross.index.tolist()

        

        # 获取个股一下个月回报
        stock_return_df = pd.DataFrame(index = stock_list)
        stock_return_list = []
        for stock in stock_list:
            # 或取下一个月度收益
            stock_close_df = pd.read_csv('D:/JUEJINStockData/'+stock+'.csv')
            return_next = stock_close_df[stock_close_df['date']==date_next]['close'].tolist()[0] / stock_close_df[stock_close_df['date']==date]['close'].tolist()[0] - 1
            stock_return_list.append(return_next)
        stock_return_df['next_return'] = stock_return_list

        # 数据拼接，构造回归用截面数据并回归
        reg_df = pd.concat([BetaCross,stock_return_df],axis=1)

        ## 去极值
        for col in reg_df.columns:
            reg_df = reg_df.sort_values(col).iloc[int(len(reg_df)*0.02):int(len(reg_df)*0.98),]

        print(reg_df.head(5))

        X = reg_df[reg_df.columns.tolist()[:-1]]
        y = reg_df[reg_df.columns.tolist()[-1]]

        X = sm.add_constant(X)
        est = sm.OLS(y,X)
        result = est.fit()
        factor_premiumn = [result.params[factor] for factor in factor_beta_list]
        factor_premiumn_df.loc[date,:] = factor_premiumn

        print('Finish:%.4f%%' %((date_cnt+1)/(len(date_list)-1)*100))
    
    # 计算月均因子溢价
    mean_vec = factor_premiumn_df.mean().tolist()
    
    # 计算T值
    N = len(factor_premiumn_df)
    std_vec = factor_premiumn_df.std().tolist()
    t_vec = list(map(lambda x1,x2:(x1/x2)*(N**0.5),mean_vec,std_vec))
    
    # 输出结果表
    save_df = pd.DataFrame(index = ['month_premiumn','t-value'],columns = factor_list)
    save_df.iloc[0,:] = mean_vec
    save_df.iloc[1,:] = t_vec

    save_df.to_csv('C:/Users/jxjsj/Desktop/JupyterHome/Data/QuantOutPut/FamaMacBethRegResult.csv')

    


    




if __name__ == '__main__':
    run(strategy_id='e96646de-82a7-11e9-bf8e-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2016-06-17 13:00:00',
        backtest_end_time='2017-08-21 15:00:00')
