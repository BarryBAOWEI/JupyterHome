library(rugarch)
library(fUnitRoots)
library(xts)
library(FinTS)
library(fitdistrplus)
library(fGarch)

# 安装包 or 从github上安装包
# install.packages('fGarch')
# library(devtools)
# install_github("cran/FinTS")
# library(FinTS)

data <- read.zoo('C:/Users/jxjsj/Desktop/JupyterHome/Data/cybz_lnr_rv_w_m_ntd_100601-190131.csv',header=TRUE,sep=',')
sample <- xts(x = data)

# 创业板指2010-06-01 - 2019-01-31日历效应

## 样本时间段调整
sample.all = sample[1:2108,]
sample.test1 = sample[1:1109,]
sample.test2 = sample[1170:1435,]
sample.test3 = sample[1416:2108,]

# 回归准备-模型设定

## 对数收益率序列的弱平稳性检验
m <- round(log(length(sample.all[,1]))) # Q(m)=ln(T)
adfTest(sample.all[,1], lags = m, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")
adfTest(sample.all[,3], lags = m, type =c("c"), title = NULL, description =NULL) # type =c("nc", "c", "ct")

## 对数收益率的Ljung-Box统计量 - 检验是否需要进行GARCH建模
m <- round(log(length(sample.all[,1]))) # Q(m)=ln(T)
lnR.mean <- mean(sample.all[,1])
at <- sample.all[,1]-lnR.mean
Box.test(at,lag=3, type='Ljung')        # Ljung-Box统计量,对数收益率的自相关,不显著则收益率不存在序列相关（仅存在弱序列相关,m>=4）
Box.test(at^2,lag=m, type='Ljung')      # Ljung-Box统计量,对数收益率平方的自相关,显著则则收益率序列不独立,均满足则适合GARCH建模

###########################################################################################
## 分布选择

### (1)正态分布-norm
spec.mod1 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(1,1), include.mean = TRUE), 
  distribution.model = "norm")
sgarch.mod1 = ugarchfit(data=sample.all[,1], spec = spec.mod1, 
                        solver = "hybrid", realizedVol = sample.all[,3])

#### 回归结果
sgarch.mod1

#### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.all[,1])))                # Q(m)=ln(T)
garch_at.mod1 = sgarch.mod1@fit[["residuals"]]
garch_sigma.mod1 = sgarch.mod1@fit[["sigma"]]
stdd_residuals.mod1 = garch_at.mod1/garch_sigma.mod1
Box.test(stdd_residuals.mod1,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.mod1^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分

#### QQ图
plot(sgarch.mod1)
norm.test <- rnorm(10000, 0, 1)
ks.test(stdd_residuals.mod1,norm.test)

#### 已实现波动与预测波动的回归分析比较：RV = a + b*sigma + u，比较R^2 & |b-1| ！不太好用！
garch_sigma.mod1 = sgarch.mod1@fit[["sigma"]]^2
RV_sigma.mod1 = data.frame(y=as.vector(sample.all[,2]),
                           var1=c(garch_sigma.mod1))
lm_RV_sigma_test.mod1 = lm(y~1+var1,data=RV_sigma.mod1)
##### R-square 越大越好
summary(lm_RV_sigma_test.mod1)$r.squared
##### |b-1| 越小越好
abs(summary(lm_RV_sigma_test.mod1)$coeff[2]-1)


### (2)学生分布-std
spec.mod2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
  distribution.model = "std")
sgarch.mod2 = ugarchfit(data=sample.all[,1], spec = spec.mod2, 
                        solver = "hybrid", realizedVol = sample.all[,3])

#### 回归结果
sgarch.mod2

#### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.all[,1])))                # Q(m)=ln(T)
garch_at.mod2 = sgarch.mod2@fit[["residuals"]]
garch_sigma.mod2 = sgarch.mod2@fit[["sigma"]]
stdd_residuals.mod2 = garch_at.mod2/garch_sigma.mod2
Box.test(stdd_residuals.mod2,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.mod2^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分

#### QQ图
plot(sgarch.mod2)
std.test <- rt(10000, 4.5)
ks.test(stdd_residuals.mod2,std.test)

#### 已实现波动与预测波动的回归分析比较：RV = a + b*sigma + u，比较R^2 & |b-1|
garch_sigma.mod2 = sgarch.mod2@fit[["sigma"]]^2
RV_sigma.mod2 = data.frame(y=as.vector(sample.all[,2]),
                           var1=c(garch_sigma.mod2))
lm_RV_sigma_test.mod2 = lm(y~1+var1,data=RV_sigma.mod2)
##### R-square 越大越好
summary(lm_RV_sigma_test.mod2)$r.squared
##### |b-1| 越小越好
abs(summary(lm_RV_sigma_test.mod2)$coeff[2]-1)


### (3)广义误差分布-ged
spec.mod3 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE), 
  distribution.model = "ged")
sgarch.mod3 = ugarchfit(data=sample.all[,1], spec = spec.mod3, 
                        solver = "hybrid", realizedVol = sample.all[,3])

#### 回归结果
sgarch.mod3

#### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.all[,1])))                # Q(m)=ln(T)
garch_at.mod3 = sgarch.mod3@fit[["residuals"]]
garch_sigma.mod3 = sgarch.mod3@fit[["sigma"]]
stdd_residuals.mod3 = garch_at.mod3/garch_sigma.mod3
Box.test(stdd_residuals.mod3,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.mod3^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分

#### QQ图
plot(sgarch.mod3)
ged.test <- rged(10000, 0, 1 ,1.2)
ks.test(stdd_residuals.mod3,ged.test)

#### 已实现波动与预测波动的回归分析比较：RV = a + b*sigma + u，比较R^2 & |b-1|
garch_sigma.mod3 = sgarch.mod3@fit[["sigma"]]^2
RV_sigma.mod3 = data.frame(y=as.vector(sample.all[,2]),
                           var1=c(garch_sigma.mod3))
lm_RV_sigma_test.mod3 = lm(y~1+var1,data=RV_sigma.mod3)
##### R-square
summary(lm_RV_sigma_test.mod3)$r.squared
##### |b-1|
abs(summary(lm_RV_sigma_test.mod3)$coeff[2]-1)
###########################################################################################

###时段1#####################################################################
### 周内效应总体检验

#### 周内效应-ged ####
spec2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(1,1), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.test1[,c(16,17,19,20)])), 
  distribution.model = "ged")
sgarch_test2 = ugarchfit(data=sample.test1[,1], spec = spec2, 
                         solver = "hybrid", realizedVol = sample.test1[,3])
sgarch_test2
plot(sgarch_test2)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test1[,1])))                # Q(m)=ln(T)
garch_at.test1 = sgarch_test2@fit[["residuals"]]
garch_sigma.test1 = sgarch_test2@fit[["sigma"]]
stdd_residuals.test1 = garch_at.test1/garch_sigma.test1
Box.test(stdd_residuals.test1,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test1^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


### 月内效应总体检验

#### 月内效应-ged
spec8 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(2,2), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.test1[,c(4:8,10:15)])), 
  distribution.model = "ged")
sgarch_test8 = ugarchfit(data=sample.test1[,1], spec = spec8, 
                         solver = "hybrid", realizedVol = sample.test1[,3])
sgarch_test8
plot(sgarch_test8)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test1[,1])))               # Q(m)=ln(T)
garch_at.test3 = sgarch_test8@fit[["residuals"]]
garch_sigma.test3 = sgarch_test8@fit[["sigma"]]
stdd_residuals.test3 = garch_at.test3/garch_sigma.test3
Box.test(stdd_residuals.test3,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test3^2,lag=m-1, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


### 假日效应总体检验

#### 假日效应-ged
spec14 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(1,1), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.test1[,c(23)])), 
  distribution.model = "ged")
sgarch_test14 = ugarchfit(data=sample.test1[,1], spec = spec14, 
                          solver = "hybrid", realizedVol = sample.test1[,3])
sgarch_test14
plot(sgarch_test14)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test1[,1])))               # Q(m)=ln(T)
garch_at.test5 = sgarch_test14@fit[["residuals"]]
garch_sigma.test5 = sgarch_test14@fit[["sigma"]]
stdd_residuals.test5 = garch_at.test5/garch_sigma.test5
Box.test(stdd_residuals.test5,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test5^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分

###时段2#####################################################################
### 周内效应总体检验

#### 周内效应-ged ####
spec2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors =as.matrix.data.frame(sample.test2[,c(16,17,19,20)])), 
  distribution.model = "ged")
sgarch_test2 = ugarchfit(data=sample.test2[,1], spec = spec2, 
                         solver = "hybrid", realizedVol = sample.test2[,3])
sgarch_test2
plot(sgarch_test2)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test2[,1])))                # Q(m)=ln(T)
garch_at.test1 = sgarch_test2@fit[["residuals"]]
garch_sigma.test1 = sgarch_test2@fit[["sigma"]]
stdd_residuals.test1 = garch_at.test1/garch_sigma.test1
Box.test(stdd_residuals.test1,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test1^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


### 月内效应总体检验

#### 月内效应-ged
spec8 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.test2[,c(4:8,10:15)])), 
  distribution.model = "ged")
sgarch_test8 = ugarchfit(data=sample.test2[,1], spec = spec8, 
                         solver = "hybrid", realizedVol = sample.test2[,3])
sgarch_test8
plot(sgarch_test8)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test2[,1])))               # Q(m)=ln(T)
garch_at.test3 = sgarch_test8@fit[["residuals"]]
garch_sigma.test3 = sgarch_test8@fit[["sigma"]]
stdd_residuals.test3 = garch_at.test3/garch_sigma.test3
Box.test(stdd_residuals.test3,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test3^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


### 假日效应总体检验

#### 假日效应-ged
spec14 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.test2[,c(23)])), 
  distribution.model = "ged")
sgarch_test14 = ugarchfit(data=sample.test2[,1], spec = spec14, 
                          solver = "hybrid", realizedVol = sample.test2[,3])
sgarch_test14
plot(sgarch_test14)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test2[,1])))               # Q(m)=ln(T)
garch_at.test5 = sgarch_test14@fit[["residuals"]]
garch_sigma.test5 = sgarch_test14@fit[["sigma"]]
stdd_residuals.test5 = garch_at.test5/garch_sigma.test5
Box.test(stdd_residuals.test5,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test5^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


###时段3#####################################################################
### 周内效应总体检验

#### 周内效应-ged ####
spec2 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.test3[,c(16,17,19,20)])), 
  distribution.model = "ged")
sgarch_test2 = ugarchfit(data=sample.test3[,1], spec = spec2, 
                         solver = "hybrid", realizedVol = sample.test3[,3])
sgarch_test2
plot(sgarch_test2)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test3[,1])))                # Q(m)=ln(T)
garch_at.test1 = sgarch_test2@fit[["residuals"]]
garch_sigma.test1 = sgarch_test2@fit[["sigma"]]
stdd_residuals.test1 = garch_at.test1/garch_sigma.test1
Box.test(stdd_residuals.test1,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test1^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


### 单星期效应检验

#### 周四效应-ged
spec5 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(4,5)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test3[,c(19)]), 
  distribution.model = "ged")
sgarch_test5 = ugarchfit(data=sample.test3[,1], spec = spec5, 
                         solver = "hybrid", realizedVol = sample.test3[,3])
sgarch_test5
plot(sgarch_test5)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test3[,1])))                # Q(m)=ln(T)
garch_at.test5 = sgarch_test5@fit[["residuals"]]
garch_sigma.test5 = sgarch_test5@fit[["sigma"]]
stdd_residuals.test5 = garch_at.test5/garch_sigma.test5
Box.test(stdd_residuals.test5,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test5^2,lag=m, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分

### 月内效应总体检验

#### 月内效应-ged
spec8 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = as.matrix.data.frame(sample.test3[,c(4:8,10:15)])), 
  distribution.model = "ged")
sgarch_test8 = ugarchfit(data=sample.test3[,1], spec = spec8, 
                         solver = "hybrid", realizedVol = sample.test3[,3])
sgarch_test8
plot(sgarch_test8)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test3[,1])))               # Q(m)=ln(T)
garch_at.test3 = sgarch_test8@fit[["residuals"]]
garch_sigma.test3 = sgarch_test8@fit[["sigma"]]
stdd_residuals.test3 = garch_at.test3/garch_sigma.test3
Box.test(stdd_residuals.test3,lag=m, type='Ljung')      # Ljung-Box统计量,标准残差的自相关,不显著则收益率方程建模充分
Box.test(stdd_residuals.test3^2,lag=m-1, type='Ljung')    # Ljung-Box统计量,标准残差平方的自相关,不显著则波动率方程建模充分


### 假日效应总体检验

#### 假日效应-ged
spec14 = ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1,1)), 
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE, external.regressors = sample.test3[,c(23)]), 
  distribution.model = "ged")
sgarch_test14 = ugarchfit(data=sample.test3[,1], spec = spec14, 
                          solver = "hybrid", realizedVol = sample.test3[,3])
sgarch_test14
plot(sgarch_test14)
##### 标准化残差的Ljung-Box统计量 - 检验建模是否充分
m <- round(log(length(sample.test3[,1])))               # Q(m)=ln(T)
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
