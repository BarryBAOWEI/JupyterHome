import numpy as np
import matplotlib.pyplot as plt

n = 100
X = np.linspace(0, 1, n).reshape(-1, 1)

def kernal_function(type):
    def f(x, y):
        return {
            0: lambda x,y: x*y,
            1: lambda x,y: min(x,y),
            2: lambda x,y: np.exp(-100*((x-y)**2))
        }[type](x,y)
    return f

kernal = kernal_function(type=1);

K = np.zeros((n, n))
for i in range(n):
    for j in range (n):
        K[i,j] = kernal(X[i], X[j])

u = np.random.randn(n, 1)
A,S,B = np.linalg.svd(K);
z = A*np.sqrt(S)*u

plt.xlim(0,1)
#plt.ylim(-2,2)
plt.plot(X, z[0], ':')