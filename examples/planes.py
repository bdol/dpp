import matplotlib.pyplot as plt
import numpy as np
import dpp.util, dpp.k_dpp

# Width, height of the sampling grid
n = 50
x, y = np.meshgrid(range(1, n+1), range(1, n+1))
x = 1/float(n)*(x.flatten())
y = 1/float(n)*(y.flatten())

# Number of samples to generate
k = 200

# Randomly sample k points
idx = np.arange(x.size)
np.random.shuffle(idx)
x_uniform = x[idx[:k]]
y_uniform = y[idx[:k]]

# Sample a k-DPP
# First construct a Gaussian L-ensemble
sigma = 0.1
L = np.exp(- ( np.power(x - x[:, None], 2) +
               np.power(y - y[:, None], 2) )/(sigma**2))

D, V = dpp.util.decompose_kernel(L)
Y = dpp.k_dpp.k_sample(k, D, V)
x_dpp = x[Y.astype(int)]
y_dpp = y[Y.astype(int)]

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(x_uniform, y_uniform, 'bo')

plt.subplot(1, 2, 2)
plt.plot(x_dpp, y_dpp, 'bo')
plt.show()
