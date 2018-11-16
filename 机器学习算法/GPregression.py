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
num_sample = 3
n = 100
#noisy_std = 0.05

# sample locations
sample_points = np.linspace(0, 1, n)

# compute covariance of the sample locations
cov_samples = cov(sample_points, sample_points)

# data points
x = np.array([0.1, 0.3, 0.8])
y = np.array([0., 0.2, -0.5])

# compute mean and covariance conditioned on the data points
cov_mix = cov(sample_points, x)
precision_data = np.linalg.inv(cov(x, x)) #If with noise: +noisy_std**2 * np.eye(len(x))
mean_cond = cov_mix @ precision_data @ y.T
cov_cond = cov_samples - cov_mix @ precision_data @ cov_mix.T

# draw sample functions and plot them
f1, f2, f3 = np.random.multivariate_normal(mean_cond, cov_cond, num_sample)
plt.plot(sample_points, f1)
plt.plot(sample_points, f2)
plt.plot(sample_points, f3)

# plot data points
plt.plot(x, y, 'k+')

# plot uncertainty region
std = np.sqrt(cov_cond.diagonal())
plt.fill_between(sample_points, mean_cond - 2 * std, mean_cond + 2 * std, facecolor='lightgray')

plt.show()