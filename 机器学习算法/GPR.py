import numpy as np
import matplotlib.pyplot as plt

def kernal_function(type):
    def f(x, y):
        return {
            0: lambda x,y: x*y,
            1: lambda x,y: min(x,y),
            2: lambda x,y: np.exp(-10*((x-y)**2)),
            3: lambda x,y: np.exp(-10*abs(x-y)),
            4: lambda x,y: min(x,y)*(1-max(x,y)),
            5: lambda x,y: min(x,y)**2*(max(x,y)/2-min(x,y)/6)
        }[type](x,y)
    return f
    
kernal = kernal_function(type=2)

def cov(x, y):
    tmp = np.zeros((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            tmp[i, j] = kernal(x[i], y[j])
    return tmp
            

# parameters
num_sample = 8
n = 400
#noisy_std = 0.05

# sample locations
sample_points = np.linspace(0., 1., n)

sample = np.zeros((n,1))
for i in range(n):
    sample[i,0] = sample_points[i]
    

# compute covariance of the sample locations
cov_samples = cov(sample, sample)

# data points
x = np.array([[0.1]
             ,[0.3]
             ,[0.8]
             ,[0.4]
             ,[0.2]
             ,[0.5]
             ,[0.7]
             ,[0.99]])
y = np.array([[0.3]
             ,[-0.2]
             ,[0.1]
             ,[-0.16]
             ,[0.04]
             ,[-0.14]
             ,[0.11]
             ,[0.2]])

# compute mean and covariance conditioned on the data points
cov_mix = cov(x, sample)

I = np.zeros((8,8))
for i in range(8):
    I[i,i] = 0.01

precision_data = np.linalg.inv(cov(x, x)+I) #If with noise: +noisy_std**2 * np.eye(len(x))
mean_cond = cov_mix.T @ precision_data @ y
# cov_cond = cov_samples - cov_mix @ precision_data @ cov_mix.T
cov_cond = cov(sample,sample) - cov_mix.T @ precision_data @ cov_mix

mean = np.zeros((1,n))
for i in range(n):
    mean[0,i] = mean_cond[i]

# draw sample functions and plot them
# f1 = np.random.multivariate_normal(mean[0], cov_cond, n)
plt.plot(sample, mean[0])



# plot data points
# plt.plot(sample_points[0], mean_cond[0], 'k+')

# plot uncertainty region
std = np.sqrt(cov_cond.diagonal())

sample_points_ss = np.zeros((1,n))
for i in range(n):
    sample_points_ss[0,i] = sample_points[i]

plt.plot(x, y, 'k+')
plt.fill_between(sample_points_ss[0], mean[0] - 2 * std, mean[0] + 2 * std, facecolor='lightgray')

plt.show()