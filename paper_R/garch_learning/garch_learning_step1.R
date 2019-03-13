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

# 安装�? or 从github上安装包
# install.packages('fGarch')
# library(devtools)
# install_github("cran/FinTS")
# library(FinTS)

# # 上证综指
data <- read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/SZZS_lnr_rv_w_m_ntd_080101-190131adj.csv',header=TRUE,sep=',')
sample <- xts(x = data)
sample.all = sample[490:2697,]
sample.test1 = sample[490:1658,]
sample.test2 = sample[1641:2004,]
sample.test3 = sample[1982:2697,]

# # 创业板指
# data <- read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/cybz_lnr_rv_w_m_ntd_100601-190131adj.csv',header=TRUE,sep=',')
# sample <- xts(x = data)
# sample.all = sample[1:2108,]
# sample.test1 = sample[1:1109,]
# sample.test2 = sample[1170:1435,]
# sample.test3 = sample[1416:2108,]
# 
# 深证成指
# data <- read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/szcz_lnr_rv_w_m_ntd_080101-190131adj.csv',header=TRUE,sep=',')
# sample <- xts(x = data)
# sample.all = sample[490:2697,]
# sample.test1 = sample[490:1779,]
# sample.test2 = sample[1759:2109,]
# sample.test3 = sample[2087:2697,]


# 回归准备-模型设定

## 对数收益率序列的弱平稳性检�?
m <- round(log(length(sample.all[,1]))) # Q(m)=ln(T)
adfTest(sample.all[,1], lags = m, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")
adfTest(sample.all[,3], lags = m, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")

## 对数收益率的Ljung-Box统计�? - 检验是否需要进行GARCH建模
m <- round(log(length(sample.all[,1]))) # Q(m)=ln(T)
lnR.mean <- mean(sample.all[,1])
at <- sample.all[,1]-lnR.mean
Box.test(at,lag=3, type='Ljung')        # Ljung-Box统计�?,对数收益率的自相�?,不显著则收益率不存在序列相关（仅存在弱序列相�?,m>=4�?
Box.test(at^2,lag=m, type='Ljung')      # Ljung-Box统计�?,对数收益率平方的自相�?,显著则则收益率序列不独立,均满足则适合GARCH建模

###########################################################################################
## 分布选择

### (1)正态分�?-norm
spec.mod1 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
  distribution.model = "norm")
sgarch.mod1 = ugarchfit(data=sample.all[,1], spec = spec.mod1, 
                        solver = "hybrid", realizedVol = sample.all[,3])

#### 回归结果
sgarch.mod1

#### 标准化残差的Ljung-Box统计�? - 检验建模是否充�?
m <- round(log(length(sample.all[,1])))                # Q(m)=ln(T)
garch_at.mod1 = sgarch.mod1@fit[["residuals"]]
garch_sigma.mod1 = sgarch.mod1@fit[["sigma"]]
stdd_residuals.mod1 = garch_at.mod1/garch_sigma.mod1
Box.test(stdd_residuals.mod1,lag=m, type='Ljung')      # Ljung-Box统计�?,标准残差的自相关,不显著则收益率方程建模充�?
Box.test(stdd_residuals.mod1^2,lag=m, type='Ljung')    # Ljung-Box统计�?,标准残差平方的自相关,不显著则波动率方程建模充�?

#### QQ�?
plot(sgarch.mod1)
norm.test <- rnorm(10000, 0, 1)
ks.test(stdd_residuals.mod1,norm.test)


### (2)学生分布-std
spec.mod2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
  distribution.model = "std")
sgarch.mod2 = ugarchfit(data=sample.all[,1], spec = spec.mod2, 
                        solver = "hybrid", realizedVol = sample.all[,3])

#### 回归结果
sgarch.mod2

#### 标准化残差的Ljung-Box统计�? - 检验建模是否充�?
m <- round(log(length(sample.all[,1])))                # Q(m)=ln(T)
garch_at.mod2 = sgarch.mod2@fit[["residuals"]]
garch_sigma.mod2 = sgarch.mod2@fit[["sigma"]]
stdd_residuals.mod2 = garch_at.mod2/garch_sigma.mod2
Box.test(stdd_residuals.mod2,lag=m, type='Ljung')      # Ljung-Box统计�?,标准残差的自相关,不显著则收益率方程建模充�?
Box.test(stdd_residuals.mod2^2,lag=m, type='Ljung')    # Ljung-Box统计�?,标准残差平方的自相关,不显著则波动率方程建模充�?

#### QQ�?
plot(sgarch.mod2)
std.test <- rt(10000, 4.5)
ks.test(stdd_residuals.mod2,std.test)


### (3)广义误差分布-ged
spec.mod3 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
  distribution.model = "ged")
sgarch.mod3 = ugarchfit(data=sample.all[,1], spec = spec.mod3, 
                        solver = "hybrid", realizedVol = sample.all[,3])

#### 回归结果
sgarch.mod3

#### 标准化残差的Ljung-Box统计�? - 检验建模是否充�?
m <- round(log(length(sample.all[,1])))                # Q(m)=ln(T)
garch_at.mod3 = sgarch.mod3@fit[["residuals"]]
garch_sigma.mod3 = sgarch.mod3@fit[["sigma"]]
stdd_residuals.mod3 = garch_at.mod3/garch_sigma.mod3
Box.test(stdd_residuals.mod3,lag=m, type='Ljung')      # Ljung-Box统计�?,标准残差的自相关,不显著则收益率方程建模充�?
Box.test(stdd_residuals.mod3^2,lag=m, type='Ljung')    # Ljung-Box统计�?,标准残差平方的自相关,不显著则波动率方程建模充�?

#### QQ�?
plot(sgarch.mod3)
ged.test <- rged(10000, 0, 1 ,1.2)
ks.test(stdd_residuals.mod3,ged.test)


###########################################################################################

###日历效应检�?#####################################################################
### 时间段选择
sample.testt = sample.test1

### 周内效应总体检�?

#### 周内效应-ged ####
spec2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(4,5)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.testt[,c(16,17,19,20)])), 
  distribution.model = "ged")
sgarch_test2 = ugarchfit(data=sample.testt[,1], spec = spec2, 
                         solver = "hybrid", realizedVol = sample.testt[,3])
sgarch_test2
plot(sgarch_test2)
##### 标准化残差的Ljung-Box统计�? - 检验建模是否充�?
m <- round(log(length(sample.testt[,1])))                # Q(m)=ln(T)
garch_at.test1 = sgarch_test2@fit[["residuals"]]
garch_sigma.test1 = sgarch_test2@fit[["sigma"]]
stdd_residuals.test1 = garch_at.test1/garch_sigma.test1
Box.test(stdd_residuals.test1,lag=m, type='Ljung')      # Ljung-Box统计�?,标准残差的自相关,不显著则收益率方程建模充�?
Box.test(stdd_residuals.test1^2,lag=m, type='Ljung')    # Ljung-Box统计�?,标准残差平方的自相关,不显著则波动率方程建模充�?


### 月内效应总体检�?

#### 月内效应-ged
spec8 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(5,5)), 
  mean.model = list(armaOrder = c(1,1), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.testt[,c(4:8,10:15)])), 
  distribution.model = "ged")
sgarch_test8 = ugarchfit(data=sample.testt[,1], spec = spec8, 
                         solver = "hybrid", realizedVol = sample.testt[,3])
sgarch_test8
plot(sgarch_test8)
##### 标准化残差的Ljung-Box统计�? - 检验建模是否充�?
m <- round(log(length(sample.testt[,1])))               # Q(m)=ln(T)
garch_at.test3 = sgarch_test8@fit[["residuals"]]
garch_sigma.test3 = sgarch_test8@fit[["sigma"]]
stdd_residuals.test3 = garch_at.test3/garch_sigma.test3
Box.test(stdd_residuals.test3,lag=m, type='Ljung')      # Ljung-Box统计�?,标准残差的自相关,不显著则收益率方程建模充�?
Box.test(stdd_residuals.test3^2,lag=m, type='Ljung')    # Ljung-Box统计�?,标准残差平方的自相关,不显著则波动率方程建模充�?


### 假日效应总体检�?

#### 假日效应-ged
spec14 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(8,8)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.testt[,c(23)])), 
  distribution.model = "ged")
sgarch_test14 = ugarchfit(data=sample.testt[,1], spec = spec14, 
                          solver = "hybrid", realizedVol = sample.testt[,3])
sgarch_test14
plot(sgarch_test14)
##### 标准化残差的Ljung-Box统计�? - 检验建模是否充�?
m <- round(log(length(sample.testt[,1])))               # Q(m)=ln(T)
garch_at.test5 = sgarch_test14@fit[["residuals"]]
garch_sigma.test5 = sgarch_test14@fit[["sigma"]]
stdd_residuals.test5 = garch_at.test5/garch_sigma.test5
Box.test(stdd_residuals.test5,lag=m, type='Ljung')      # Ljung-Box统计�?,标准残差的自相关,不显著则收益率方程建模充�?
Box.test(stdd_residuals.test5^2,lag=m, type='Ljung')    # Ljung-Box统计�?,标准残差平方的自相关,不显著则波动率方程建模充�?