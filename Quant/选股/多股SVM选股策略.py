# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from datetime import datetime
import numpy as np
from gm.api import *
import sys
try:
    from sklearn import svm
except:
    print('请安装scikit-learn库和带mkl的numpy')
    sys.exit(-1)

'''
多股票基于SVM的择股模型，
训练特征可以修改，模型也可嵌套入其他策略中，
掘金3量化终端可运行，需要修改strategy_id与token，
每周一开盘时运行一次，运用过去一段训练时间窗口的数据进行移动窗口采样生成训练数据，目标值y为训练时间窗口后一段窗口的涨跌情况，
模型训练完毕后，读取至前一天为止长度为1个训练时间窗口的数据，生成预测用数据，输入模型得到输出0或1，对多个股票重复训练与预测过程，平均买入输出为1的股票，
每次交易，当期预测结果为1则调仓至1/N，当期预测结果为0则调仓至0。
'''

def init(context):
    # 定时执行任务
    schedule(schedule_func=algo, date_rule='1w', time_rule='09:31:00')
    # context.symbol = 'SZSE.000651'
    # context.symbols = get_constituents('SHSE.000016', fields=None, df=False)
    
    context.Xwind = 15
    context.Ywind = 5
    context.trainwind = 80

def algo(context):

    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    # weekday = context.now.isoweekday()

    # 获取成分股列表
    # print(get_history_constituents(index='SHSE.000016', start_date=last_day, end_date=last_day))
    # context.symbols_list = get_history_constituents(index='SHSE.000016', start_date=last_day, end_date=last_day)[0]['constituents'].keys()
    context.symbols_list = get_constituents(index='SHSE.000016')
    context.symbols_len =len(context.symbols_list)

    # 新的一周开始重新训练模型，更新参数
    # if weekday == 1:

    print(context.now)
    
    # 礼拜一全清仓，需改写，因为连续预测为买入无需清仓节约费用
    # order_close_all()

    cnt = 0
    prediction_all = []
    for context.symbol in context.symbols_list:
        cnt += 1
        end_date = last_day  # SVM训练终止时间
        probability = False  # False输出0,1，True输出概率
        # 获取目标股票的daily历史行情
        recent_data = history_n(context.symbol, frequency='1d', count=context.trainwind, end_time=end_date, fill_missing='last',
                            df=True)
        
        # 无数据，本期剔除
        try:
            days_value = recent_data['bob'].values
            days_close = recent_data['close'].values
        except:
            prediction_all.append(0)
            continue
        # 数据不足，本期剔除
        if len(recent_data) != context.trainwind:
            prediction_all.append(0)
            continue
        
        days = []
        # 获取行情日期列表
        # print('准备数据训练SVM')
        for i in range(len(days_value)):
            days.append(str(days_value[i])[0:10])
        x_all = []
        y_all = []
        # 采样窗口滑动范围
        for index in range(context.Xwind, (len(days) - context.Ywind)):
            # 取样
            data = recent_data.iloc[(index - context.Xwind):index,:]
            dataForCompute = data.copy()
            dataForCompute['volume'] = dataForCompute.amount/dataForCompute.close

            close = dataForCompute['close'].values
            max_x = dataForCompute['high'].values
            min_n = dataForCompute['low'].values
            amount = dataForCompute['amount'].values
            volume = dataForCompute['volume'].values
            
            # 趋势指标
            close_ratio = close[-1] / np.mean(close)  # 收盘价/均值
            volume_ratio = volume[-1] / np.mean(volume)  # 现量/均量
            max_ratio = max_x[-1] / np.mean(max_x)  # 最高价/均价
            min_ratio = min_n[-1] / np.mean(min_n)  # 最低价/均价
            return_now = close[-1] / close[0]  # 区间收益率
            
            # 均量指标
            close_mean = np.mean(close)  # 收盘价均值
            volume_mean = np.mean(volume)  # 现量均量
            max_mean = np.mean(max_x)  # 最高价均价
            min_mean = np.mean(min_n)  # 最低价均价
            vol = volume[-1]  # 现量
            std = np.std(np.array(close), axis=0)  # 区间标准差

            # 将计算出的指标添加到训练集X
            # features用于存放因子
            # features = [close_ratio, volume_ratio, max_ratio, min_ratio, return_now, vol, std]
            features = [close_ratio, volume_ratio, max_ratio, min_ratio, return_now, 
                        close_mean, volume_mean, max_mean, min_mean, vol, std]
            x_all.append(features)
        # 准备算法需要用到的数据
        for i in range(len(days_close) - (context.Xwind+context.Ywind)):
            if days_close[i + (context.Xwind+context.Ywind)] > days_close[i + context.Xwind]:
                label = 1
            else:
                label = 0
            y_all.append(label)
        x_train = x_all[: -1]
        y_train = y_all[: -1]
        # 训练SVM - 线性核或者高斯核
        context.clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=probability,
                            tol=0.001, cache_size=200, verbose=False, max_iter=-1,
                            decision_function_shape='ovr', random_state=None)      
        context.clf.fit(x_train, y_train)
        # print('训练完成%.4f' %(cnt/context.symbols_len),end='\r')

    # 获取模型相关的数据
    # 获取持仓
    # position = context.account().position(symbol=context.symbol, side=PositionSide_Long)

        # 获取预测用的历史数据
        data = history_n(symbol=context.symbol, frequency='1d', end_time=last_day, count=context.Xwind,
                        fill_missing='last', df=True)
        dataForCompute = data.copy()
        dataForCompute['volume'] = dataForCompute.amount/dataForCompute.close
        
        close = dataForCompute['close'].values
        max_x = dataForCompute['high'].values
        min_n = dataForCompute['low'].values
        amount = dataForCompute['amount'].values
        volume = dataForCompute['volume'].values
        
        # 趋势指标
        close_ratio = close[-1] / np.mean(close)  # 收盘价/均值
        volume_ratio = volume[-1] / np.mean(volume)  # 现量/均量
        max_ratio = max_x[-1] / np.mean(max_x)  # 最高价/均价
        min_ratio = min_n[-1] / np.mean(min_n)  # 最低价/均价
        return_now = close[-1] / close[0]  # 区间收益率
        
        # 均量指标
        close_mean = np.mean(close)  # 收盘价均值
        volume_mean = np.mean(volume)  # 现量均量
        max_mean = np.mean(max_x)  # 最高价均价
        min_mean = np.mean(min_n)  # 最低价均价
        vol = volume[-1]  # 现量
        std = np.std(np.array(close), axis=0)  # 区间标准差

        # 将计算出的指标添加到训练集X
        # features用于存放因子
        # features = [close_ratio, volume_ratio, max_ratio, min_ratio, return_now, vol, std]
        features = [close_ratio, volume_ratio, max_ratio, min_ratio, return_now, 
                    close_mean, volume_mean, max_mean, min_mean, vol, std]

        # 得到本次输入模型的因子，是一次观测
        features = np.array(features).reshape(1, -1)
        prediction = context.clf.predict(features)[0]
        prediction_all.append(prediction)
        # print('预测结果:%d' %(prediction))
        
    # 若预测值为上涨则开仓或保持仓位,否则清仓
    buy_sum = sum(prediction_all)
    if buy_sum != 0:
        for each_order in range(context.symbols_len):
            predict_result = prediction_all[each_order]
            symbol_result = context.symbols_list[each_order]
            if predict_result == 1:
                order_target_percent(symbol=symbol_result, percent=0.95/buy_sum, order_type=OrderType_Market,
                                position_side=PositionSide_Long)
            else:
                order_target_percent(symbol=symbol_result, percent=0, order_type=OrderType_Market,
                                position_side=PositionSide_Long)

if __name__ == '__main__':
    run(strategy_id='13e1efb6-7d60-11e9-a93a-b025aa2961ed',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='f69f85e5e8f97fab3dda4e3641dc722acca1c2e0',
        backtest_start_time='2011-01-01 15:30:00',
        backtest_end_time='2012-01-22 15:30:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)