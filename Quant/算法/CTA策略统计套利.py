# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
from gm.api import 
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
try
    import statsmodels.tsa.stattools as ts
except
    print('请安装statsmodels库')

'''
本策略根据EG两步法(1.序列同阶单整2.OLS残差平稳)判断序列具有协整关系之后(若无协整关系则全平仓位不进行操作)
通过计算两个真实价格序列回归残差的0.9个标准差上下轨,并在价差突破上轨的时候做空价差,价差突破下轨的时候做多价差
并在回归至标准差水平内的时候平仓
回测数据为SHFE.rb1801和SHFE.rb1805的1min数据
回测时间为2017-09-25 080000到2017-10-01 150000

问题1：为什么上下轨是标准差0.9倍，依据？ - 存在BUG，价差序列非正态，始终存在一定偏度，往某一方向移动的概率较大，则给该方向的轨道变严格（上轨则往上，下轨则往下），另一方向宽松。
问题2：跨期商品价差过程是否均值回复过程？ - 通过协整检验，必然说明历史数据的残差序列具有均值回复特性。
漏洞3：上穿上轨，建立价差空仓，价差没有立即回复，则均线&上轨&下轨会不断上移，倘若均值发生变化，等下穿上轨回复时再平仓，发生亏损。 
- 价差历史数据是多峰数据，确实在多个均值附近波动，协整检验会排除掉价差开始明显偏离均值的过程，但是在偏离初期难以识别（还是统计上协整的）。
- 降低数据窗口可增强对近期数据的敏感性，有效识别均值移动，但是会使策略失效 - 上穿上轨空仓建立，窗口短上轨移动快，等下穿下轨的时候价格是上升的，亏损。
- 使协整检验更加严格，可行，但大量减少交易次数，会错失机会 - 这不是大问题，因为可以加入更多标的pairs来增加交易机会。
漏洞4：交易及其频繁，手续费极高，交易胜率低。
'''

def init(context)
    # 统计套利候选 - 单组跨期
    # context.goods = ['SHFE.rb1801', 'SHFE.rb1805']
    # 统计套利候选 - 多组跨期
    context.kinds = ['rb','cu','al','au','fu','bu','zn','pb','sn','ni','hc']
    context.interperiod = ['1801','1805']
    context.goods_list = [['SHFE.'+i+j for j in context.interperiod] for i in context.kinds]
    context.goods = list(np.array(context.goods_list).reshape(-1))
    # 显著性阈值
    context.sig = 0.001
    # 交易合约手
    context.share = 100
    # 订阅品种
    subscribe(symbols=context.goods, frequency='60s', count=801, wait_group=True)
    # 保存一下pvalue序列
    # context.pvalue_series = []
def on_backtest_finished(context, indicator)
    # plt.plot(context.pvalue_series)
    # plt.show()
    print(indicator)
def on_bar(context, bars)
    # for goods in context.goods_list
    goods = ['SHFE.rb1801', 'SHFE.rb1805']
    # 获取过去800个60s的收盘价数据
    close_01 = context.data(symbol=goods[0], frequency='60s', count=801, fields='close')['close'].values
    close_02 = context.data(symbol=goods[1], frequency='60s', count=801, fields='close')['close'].values
    # 无交易则剔除
    # if len(close_01) != 801 or len(close_02) != 801
    #     continue
    # 展示两个价格序列的协整检验的结果
    pvalue = sm.tsa.stattools.coint(close_01, close_02)[1]
    # context.pvalue_series.append(pvalue)
    # 如果返回协整检验不通过的结果则全平仓位等待
    if pvalue=context.sig
        print('协整检验不通过,全平所有仓位')
        order_close_all()
        return
    # 回归获得残差序列与系数
    y = close_01
    X = close_02
    X = sm.add_constant(X)
    est = sm.OLS(y,X)
    result = est.fit()
    # constant = list(result.params)[0]
    beta = list(result.params)[1]
    beta_round = int(round(betacontext.share))
    resid = result.resid
    resid_df = pd.DataFrame(resid)
    # plt.plot(resid)
    # plt.plot(resid_df.rolling(700).mean()+resid_df.rolling(700).std())
    # plt.plot(resid_df.rolling(700).mean()-resid_df.rolling(700).std())
    # plt.show()
    # 计算残差的标准差上下轨
    mean = np.mean(resid)
    up = mean + 0.9  np.std(resid)
    down = mean - 0.9  np.std(resid)
    # skew = resid_df.skew()[0]
    # # 惩罚项，偏度绝对值在0-1时惩罚较小，大于1时惩罚开始快速增强
    # punish = skew2
    # if skew  0
    #     up = mean + 0.9  np.std(resid)  (2 + punish)2
    #     down = mean - 0.9  np.std(resid)  (2 - punish)2
    # else
    #     up = mean + 0.9  np.std(resid)  (2 - punish)2
    #     down = mean - 0.9  np.std(resid)  (2 + punish)2
    # 计算新残差
    resid_new = resid[-1]
    # 获取rb1801的多空仓位
    position_01_long = context.account().position(symbol=goods[0], side=PositionSide_Long)
    position_01_short = context.account().position(symbol=goods[0], side=PositionSide_Short)
    
    # 策略编写，协整关系变量回归残差具有均值回归特性
    # 无仓位时，上穿或者下穿建仓
    if not position_01_long and not position_01_short
        # 上穿上轨时做空新残差
        if resid_new  up
            order_target_volume(symbol=goods[0], volume=context.share, order_type=OrderType_Market,
                                position_side=PositionSide_Short)
            print(goods[0] + '以市价单开空仓%d手' %(context.share))
            order_target_volume(symbol=goods[1], volume=beta_round, order_type=OrderType_Market,
                                position_side=PositionSide_Long)
            print(goods[1] + '以市价单开多仓%d手' %(beta_round))
        # 下穿下轨时做多新残差
        if resid_new  down
            order_target_volume(symbol=goods[0], volume=context.share, order_type=OrderType_Market,
                                position_side=PositionSide_Long)
            print(goods[0] + '以市价单开多仓%d手' %(context.share))
            order_target_volume(symbol=goods[1], volume=beta_round, order_type=OrderType_Market,
                                position_side=PositionSide_Short)
            print(goods[1] + '以市价单开空仓%d手' %(beta_round))
    # 新残差回归时平仓
    # 做空残差时，下穿上轨就平仓，再穿下轨继续反向建仓
    elif position_01_short
        if resid_new = up
            order_close_all()
            print('价格回归,平掉所有仓位')
        # 突破下轨反向开仓
        if resid_new  down
            order_target_volume(symbol=goods[0], volume=context.share, order_type=OrderType_Market,
                                position_side=PositionSide_Long)
            print(goods[0] + '以市价单开多仓%d手' %(context.share))
            order_target_volume(symbol=goods[1], volume=beta_round, order_type=OrderType_Market,
                                position_side=PositionSide_Short)
            print(goods[1] + '以市价单开空仓%d手' %(beta_round))
    # 做多残差时，上穿下轨就平仓，再穿上轨继续反向建仓
    elif position_01_long
        if resid_new = down
            order_close_all()
            print('价格回归,平所有仓位')
        # 突破上轨反向开仓
        if resid_new  up
            order_target_volume(symbol=goods[0], volume=context.share, order_type=OrderType_Market,
                                position_side=PositionSide_Short)
            print(goods[0] + '以市价单开空仓%d手' %(context.share))
            order_target_volume(symbol=goods[1], volume=beta_round, order_type=OrderType_Market,
                                position_side=PositionSide_Long)
            print(goods[1] + '以市价单开多仓%d手' %(beta_round))

if __name__ == '__main__'
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式MODE_LIVE回测模式MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权ADJUST_NONE前复权ADJUST_PREV后复权ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='ee3be3e4-7e16-11e9-a75e-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2017-07-04 080000',
        backtest_end_time='2017-07-30 160000',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=50000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)