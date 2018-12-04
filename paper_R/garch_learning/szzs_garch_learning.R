library(rugarch)
library(xts)
library(FinTS)

# 安装包 or 从github上安装包
# install.packages('devtools')
# library(devtools)
# install_github("cran/FinTS")
# library(FinTS)

data<-read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/SZZS_lnr_rv_w_m_ntd_080101-181101.csv',header=TRUE,sep=',')
sample <- xts(x = data)

# 上证指数2008-01-03 - 2018-11-01日历效应

## 全样本时间段

### 周内效应总体检验

#### 周内效应-norm
spec1 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(16,17,19,20)]), 
  distribution.model = "norm")
sgarch_test1 = ugarchfit(data=sample[,1], spec = spec1, 
                        solver = "hybrid", realizedVol = sample[,3])
sgarch_test1
plot(sgarch_test1)

#### 周内效应-ged
spec2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(16,17,19,20)]), 
  distribution.model = "ged")
sgarch_test2 = ugarchfit(data=sample[,1], spec = spec2, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test2
plot(sgarch_test2)

#### 周内效应-std
spec4 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(16,17,19,20)]), 
  distribution.model = "std")
sgarch_test4 = ugarchfit(data=sample[,1], spec = spec4, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test4
plot(sgarch_test4)



### 单星期效应检验

#### 周四效应-norm
spec3 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(19)]), 
  distribution.model = "norm")
sgarch_test3 = ugarchfit(data=sample[,1], spec = spec3, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test3
plot(sgarch_test3)

#### 周四效应-ged
spec5 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(19)]), 
  distribution.model = "ged")
sgarch_test5 = ugarchfit(data=sample[,1], spec = spec5, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test5
plot(sgarch_test5)

#### 周四效应-std
spec6 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1), external.regressors = sample[,c(19)]), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(19)]), 
  distribution.model = "std")
sgarch_test6 = ugarchfit(data=sample[,1], spec = spec6, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test6
plot(sgarch_test6)



### 月内效应总体检验

#### 月内效应-norm
spec7 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(4:8,10:15)]), 
  distribution.model = "norm")
sgarch_test7 = ugarchfit(data=sample[,1], spec = spec7, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test7
plot(sgarch_test7)

#### 月内效应-ged
spec8 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(4:8,10:15)]), 
  distribution.model = "ged")
sgarch_test8 = ugarchfit(data=sample[,1], spec = spec8, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test8
plot(sgarch_test8)

#### 月内效应-std
spec9 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(4:8,10:15)]), 
  distribution.model = "std")
sgarch_test9 = ugarchfit(data=sample[,1], spec = spec9, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test9
plot(sgarch_test9)



### 单月份效应检验

#### 二六七十效应-norm
spec10 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(5,9,10,13)]), 
  distribution.model = "norm")
sgarch_test10 = ugarchfit(data=sample[,1], spec = spec10, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test10
plot(sgarch_test10)

#### 二六效应-norm
spec11 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(5,9)]), 
  distribution.model = "norm")
sgarch_test11 = ugarchfit(data=sample[,1], spec = spec11, 
                          solver = "hybrid", realizedVol = sample[,3])
sgarch_test11
plot(sgarch_test11)

#### 二六效应-ged
spec12 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(5,9)]), 
  distribution.model = "ged")
sgarch_test12 = ugarchfit(data=sample[,1], spec = spec12, 
                          solver = "hybrid", realizedVol = sample[,3])
sgarch_test12
plot(sgarch_test12)

#### 二六效应-std
spec13 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(5,9)]), 
  distribution.model = "std")
sgarch_test13 = ugarchfit(data=sample[,1], spec = spec13, 
                          solver = "hybrid", realizedVol = sample[,3])
sgarch_test13
plot(sgarch_test13)



### 假日效应总体检验

#### 假日效应-norm
spec14 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample[,c(23)]), 
  distribution.model = "norm")
sgarch_test14 = ugarchfit(data=sample[,1], spec = spec14, 
                         solver = "hybrid", realizedVol = sample[,3])
sgarch_test14
plot(sgarch_test14)

#### 假日效应-norm-置于波动率方程
spec15 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1), external.regressors = sample[,c(23)]), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
  distribution.model = "norm")
sgarch_test15 = ugarchfit(data=sample[,1], spec = spec15, 
                          solver = "hybrid", realizedVol = sample[,3])
sgarch_test15
plot(sgarch_test15)

########################各类检验##########################
# arch效应检验,yt = u + at, at = yt - u, 对{at^2}序列计算LM统计量, 原假设对滞后项回归系数联合为0
ArchTest(sample[,1],lag=12)

# Ljung-Box检验，原假设序列的前n各自相关函数联合为0
Box.test((sample[,1]-mean(sample[,1]))^2,lag=12, type='Ljung')
Box.test(sample[,1],lag=3, type='Ljung')

# 标准化残差的arch效应检验LM统计量与Ljung-Box统计量
garch_at15 = sgarch_test15@fit[["residuals"]]
garch_sigma15 = sgarch_test15@fit[["sigma"]]
stdd.residuals15 = garch_at15/garch_sigma15
ArchTest(stdd.residuals15,lag=12)                    # LM统计量
Box.test(stdd.residuals15,lag=12, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,收益率方程建模充分
Box.test(stdd.residuals15^2,lag=12, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,波动率方程建模充分


########################乱七八糟##########################
sample[,c(23)]
plot(sgarch_test)

garch_at = sgarch_test@fit[["residuals"]]
garch_sigmat = sgarch_test@fit[["sigma"]]

# 'Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'
# 'M','Tu','W','Th',"F"

sample[,c(16,17,18,19,20)]
sample[,c(4:15)]
