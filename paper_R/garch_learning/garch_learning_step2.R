remove.packages()

library(timeDate)
library(timeSeries)
library(fBasics)
library(zoo)
library(rugarch)
library(fUnitRoots)
library(xts)
library(FinTS)
# library(fitdistrplus)
library(fGarch)

# 安装包 or 从github上安装包
# install.packages('fGarch')
# library(devtools)
# install_github("cran/FinTS")
# library(FinTS)

# # 上证综指
# data <- read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/SZZS_lnr_rv_w_m_ntd_080101-190131adj.csv',header=TRUE,sep=',')
# sample <- xts(x = data)
# sample.all = sample[490:2697,]
# sample.test1 = sample[490:1658,]
# sample.test2 = sample[1641:2004,]
# sample.test3 = sample[1982:2697,]

# # 创业板指
data <- read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/cybz_lnr_rv_w_m_ntd_100601-190131adj.csv',header=TRUE,sep=',')
sample <- xts(x = data)
sample.all = sample[1:2108,]
sample.test1 = sample[1:1109,]
sample.test2 = sample[1170:1435,]
sample.test3 = sample[1416:2108,]

# 深证成指
# data <- read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/szcz_lnr_rv_w_m_ntd_080101-190131adj.csv',header=TRUE,sep=',')
# sample <- xts(x = data)
# sample.all = sample[490:2697,]
# sample.test1 = sample[490:1779,]
# sample.test2 = sample[1759:2109,]
# sample.test3 = sample[2087:2697,]

## 影响因素平稳性检验ADF-test
m <- round(log(length(sample.all[,1]))) # Q(m)=ln(T)
adfTest(sample.all[,51], lags = m, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")
adfTest(sample.all[,52], lags = m, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")
adfTest(sample.all[,53], lags = m, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")


###日历效应影响因素检验#####################################################################

### 周历效应

### 时间段选择
sample.testc = sample.test3

#### 周内效应-ged ####
spec2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors =as.matrix.data.frame(sample.testc[,c(16,17,19,20,43)])), 
  distribution.model = "ged")
sgarch_test2 = ugarchfit(data=sample.testc[,1], spec = spec2, 
                         solver = "hybrid", realizedVol = sample.testc[,3])
sgarch_test2
plot(sgarch_test2)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.testc[,1])))                # Q(m)=ln(T)
garch_at.test1 = sgarch_test2@fit[["residuals"]]
garch_sigma.test1 = sgarch_test2@fit[["sigma"]]
stdd_residuals.test1 = garch_at.test1/garch_sigma.test1
Box.test(stdd_residuals.test1,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test1^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分

##### 提取交互项系数
sgarch_test2@fit[['solver']]$`sol`$`pars`[['mxreg5']]


### 月历效应

### 时间段选择
sample.testc = sample.test3

#### 月内效应-ged
spec8 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.testc[,c(4:8,10:15,49)])), 
  distribution.model = "ged")
sgarch_test8 = ugarchfit(data=sample.testc[,1], spec = spec8, 
                         solver = "hybrid", realizedVol = sample.testc[,3])
sgarch_test8
plot(sgarch_test8)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.testc[,1])))               # Q(m)=ln(T)
garch_at.test3 = sgarch_test8@fit[["residuals"]]
garch_sigma.test3 = sgarch_test8@fit[["sigma"]]
stdd_residuals.test3 = garch_at.test3/garch_sigma.test3
Box.test(stdd_residuals.test3,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test3^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


### 假日效应

### 时间段选择
sample.testc = sample.test2

#### 假日效应-ged
spec14 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.testc[,c(23,50)]), 
  distribution.model = "ged")
sgarch_test14 = ugarchfit(data=sample.testc[,1], spec = spec14, 
                          solver = "hybrid", realizedVol = sample.testc[,3])
sgarch_test14
plot(sgarch_test14)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.testc[,1])))               # Q(m)=ln(T)
garch_at.test5 = sgarch_test14@fit[["residuals"]]
garch_sigma.test5 = sgarch_test14@fit[["sigma"]]
stdd_residuals.test5 = garch_at.test5/garch_sigma.test5
Box.test(stdd_residuals.test5,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test5^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分



########################各类检验##########################
#  对数收益率序列的弱平稳性检验
adfTest(sample.test[,1], lags = 10, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")

# 寻找对数收益率序列最合适的分布-1
# # 正态分布
# fitdist(as.vector(sample.test[,1]), 'norm', method = c("mle")) # start=NULL, fix.arg=NULL, discrete=FALSE, keepdata = TRUE, keepdata.nb=100, ...
# ks.test(as.vector(sample.test[,1]), 'pnorm')
# # 广义误差分布
# ks.test(as.vector(sample.test[,1]), pt(length(as.vector(sample.test[,1])),2))
# ks.test(as.vector(sample.test[,1]), pged())

# 寻找对数收益率序列最合适的分布-2
# 已实现波动与预测波动的回归分析比较
garch_sigma = sgarch_test1@fit[["sigma"]]^2
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
garch_at1 = sgarch_test1@fit[["residuals"]]
garch_sigma1 = sgarch_test1@fit[["sigma"]]
stdd.residuals1 = garch_at1/garch_sigma1
ArchTest(stdd.residuals1,lag=7)                    # LM统计量
Box.test(stdd.residuals1,lag=7, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,收益率方程建模充分
Box.test(stdd.residuals1^2,lag=7, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,波动率方程建模充分


########################乱七八糟##########################
sample[1966:2635,c(23)]
plot(sgarch_test)

garch_at = sgarch_test@fit[["residuals"]]
garch_sigmat = sgarch_test@fit[["sigma"]]

# 'Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'
# 'M','Tu','W','Th',"F"

sample[,c(16,17,18,19,20)]
sample[,c(4:15)]
