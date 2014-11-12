import matplotlib.pyplot as plt
import numpy as np
import dpp.util, dpp.k_dpp
import sbs.sbs
from datetime import datetime


# Width, height of the sampling grid
n = 50
x, y = np.meshgrid(range(1, n+1), range(1, n+1))
x = 1/float(n)*(x.flatten())
y = 1/float(n)*(y.flatten())

# Number of samples to generate
k = 300

# Randomly sample k points
idx = np.arange(x.size)
np.random.shuffle(idx)
x_uniform = x[idx[:k]]
y_uniform = y[idx[:k]]

# First construct a Gaussian L-ensemble
sigma = 0.05
L = np.exp(- ( np.power(x - x[:, None], 2) +
               np.power(y - y[:, None], 2) )/(sigma**2))

# SBS sample
before = datetime.now()
Y = sbs.sbs.simbased_sample(k, L)
x_sbs = x[Y.astype(int)]
y_sbs = y[Y.astype(int)]
after = datetime.now()
print "SBS took {0} s.".format((after-before).total_seconds())

# Sample a k-DPP
before = datetime.now()
D, V = dpp.util.decompose_kernel(L)
Y = dpp.k_dpp.k_sample(k, D, V)
x_dpp = x[Y.astype(int)]
y_dpp = y[Y.astype(int)]
after = datetime.now()
print "DPP took {0} s.".format((after-before).total_seconds())

plt.figure(1)
plt.subplot(1, 3, 1)
plt.plot(x_uniform, y_uniform, 'bo')

plt.subplot(1, 3, 2)
plt.plot(x_dpp, y_dpp, 'bo')

plt.subplot(1, 3, 3)
plt.plot(x_sbs, y_sbs, 'bo')
plt.show()
