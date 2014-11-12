import numpy as np

def logsumexp(x):
    y = np.max(x)
    if np.isinf(y):
        return y

    x_s = x-y
    s = y + np.log(np.sum(np.exp(x_s)))

    return s
