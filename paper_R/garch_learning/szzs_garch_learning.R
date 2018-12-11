library(rugarch)
library(fUnitRoots)
library(xts)
library(FinTS)
library(fitdistrplus)

# 安装包 or 从github上安装包
# install.packages('devtools')
# library(devtools)
# install_github("cran/FinTS")
# library(FinTS)

data<-read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/SZZS_lnr_rv_w_m_ntd_080101-181101.csv',header=TRUE,sep=',')
sample <- xts(x = data)

# 上证指数2008-01-03 - 2018-11-01日历效应

## 样本时间段调整
## 2010.01初-2014.12末[489:1701],
## 2014.11初-2016.03末[1659:2004],
## 2016.02初-2018.11初*[1966:2635].
sample.test = sample[489:1701,]

### 周内效应总体检验

#### 周内效应-norm
spec1 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(2,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(16,17,19,20)]), 
  distribution.model = "norm")
sgarch_test1 = ugarchfit(data=sample.test[,1], spec = spec1, 
                        solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test1
plot(sgarch_test1)

#### 周内效应-ged
spec2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(2,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(16,17,19,20)]), 
  distribution.model = "ged")
sgarch_test2 = ugarchfit(data=sample.test[,1], spec = spec2, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test2
plot(sgarch_test2)

#### 周内效应-std
spec4 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(2,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(16,17,19,20)]), 
  distribution.model = "std")
sgarch_test4 = ugarchfit(data=sample.test[,1], spec = spec4, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test4
plot(sgarch_test4)



### 单星期效应检验

#### 周四、五效应-norm
spec3 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(2,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(19,20)]), 
  distribution.model = "norm")
sgarch_test3 = ugarchfit(data=sample.test[,1], spec = spec3, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test3
plot(sgarch_test3)

#### 周四、五效应-ged
spec5 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(2,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(19,20)]), 
  distribution.model = "ged")
sgarch_test5 = ugarchfit(data=sample.test[,1], spec = spec5, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test5
plot(sgarch_test5)

#### 周四、五效应-std
spec6 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(2,1), external.regressors = sample.test[,c(19,20)]), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(19)]), 
  distribution.model = "std")
sgarch_test6 = ugarchfit(data=sample.test[,1], spec = spec6, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test6
plot(sgarch_test6)



### 月内效应总体检验

#### 月内效应-norm
spec7 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(4:8,10:15)]), 
  distribution.model = "norm")
sgarch_test7 = ugarchfit(data=sample.test[,1], spec = spec7, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test7
plot(sgarch_test7)

#### 月内效应-ged
spec8 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(4:8,10:15)]), 
  distribution.model = "ged")
sgarch_test8 = ugarchfit(data=sample.test[,1], spec = spec8, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test8
plot(sgarch_test8)

#### 月内效应-std
spec9 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(4:8,10:15)]), 
  distribution.model = "std")
sgarch_test9 = ugarchfit(data=sample.test[,1], spec = spec9, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test9
plot(sgarch_test9)



### 单月份效应检验

#### 二六七十效应-norm
spec10 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(5,9,10,13)]), 
  distribution.model = "norm")
sgarch_test10 = ugarchfit(data=sample.test[,1], spec = spec10, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test10
plot(sgarch_test10)

#### 二六效应-norm
spec11 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(5,9)]), 
  distribution.model = "norm")
sgarch_test11 = ugarchfit(data=sample.test[,1], spec = spec11, 
                          solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test11
plot(sgarch_test11)

#### 二六效应-ged
spec12 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(5,9)]), 
  distribution.model = "ged")
sgarch_test12 = ugarchfit(data=sample.test[,1], spec = spec12, 
                          solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test12
plot(sgarch_test12)

#### 二六效应-std
spec13 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(5,9)]), 
  distribution.model = "std")
sgarch_test13 = ugarchfit(data=sample.test[,1], spec = spec13, 
                          solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test13
plot(sgarch_test13)



### 假日效应总体检验

#### 假日效应-norm
spec14 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test[,c(23)]), 
  distribution.model = "norm")
sgarch_test14 = ugarchfit(data=sample.test[,1], spec = spec14, 
                         solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test14
plot(sgarch_test14)

#### 假日效应-norm-置于波动率方程
spec15 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1), external.regressors = sample.test[,c(23)]), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
  distribution.model = "norm")
sgarch_test15 = ugarchfit(data=sample.test[,1], spec = spec15, 
                          solver = "hybrid", realizedVol = sample.test[,3])
sgarch_test15
plot(sgarch_test15)

########################各类检验##########################
#  对数收益率序列的弱平稳性检验
adfTest(sample.test[,1], lags = 1, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")

# 寻找对数收益率序列最合适的分布-1
# # 正态分布
# fitdist(as.vector(sample.test[,1]), 'norm', method = c("mle")) # start=NULL, fix.arg=NULL, discrete=FALSE, keepdata = TRUE, keepdata.nb=100, ...
# ks.test(as.vector(sample.test[,1]), 'pnorm')
# # 广义误差分布
# ks.test(as.vector(sample.test[,1]), pt(length(as.vector(sample.test[,1])),2))
# ks.test(as.vector(sample.test[,1]), pged())

# 寻找对数收益率序列最合适的分布-2
# 已实现波动与预测波动的回归分析比较
garch_sigma = sgarch_test4@fit[["sigma"]]^2
RV_sigma = data.frame(y=as.vector(sample.test[,2]),
                      var1=c(garch_sigma))
lm.RV_sigma_test = lm(y~1+var1,data=RV_sigma)
summary(lm.RV_sigma_test)
# 已实现波动与预测波动的均方误差
sum((as.vector(sample.test[,2])-garch_sigma)^2)


# arch效应检验,yt = u + at, at = yt - u, 对{at^2}序列计算LM统计量, 原假设对滞后项回归系数联合为0
ArchTest(sample.test[,1],lag=12)

# Ljung-Box检验，原假设序列的前n各自相关函数联合为0
Box.test((sample.test[,1]-mean(sample.test[,1]))^2,lag=12, type='Ljung')
Box.test(sample.test[,1],lag=3, type='Ljung')

# 标准化残差的arch效应检验LM统计量与Ljung-Box统计量
garch_at3 = sgarch_test3@fit[["residuals"]]
garch_sigma3 = sgarch_test3@fit[["sigma"]]
stdd.residuals3 = garch_at3/garch_sigma3
ArchTest(stdd.residuals3,lag=7)                    # LM统计量
Box.test(stdd.residuals3,lag=7, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,收益率方程建模充分
Box.test(stdd.residuals3^2,lag=7, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,波动率方程建模充分


########################乱七八糟##########################
sample[1966:2635,c(23)]
plot(sgarch_test)

garch_at = sgarch_test@fit[["residuals"]]
garch_sigmat = sgarch_test@fit[["sigma"]]

# 'Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'
# 'M','Tu','W','Th',"F"

sample[,c(16,17,18,19,20)]
sample[,c(4:15)]
