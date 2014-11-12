import numpy as np
from util import logsumexp

def simbased_sample(k, L):
    N = L.shape[0]
    D = np.real(np.log(1-L))

    # Sample a random element
    j = np.random.randint(N)
    Y = [j]

    for i in range(1, k):
        # Calculate distances
        p = np.sum(D[:, Y], axis=1)

        # Normalize
        p = np.exp(p - logsumexp(p))

        # Sample
        p_cdf = np.cumsum(p)
        r = np.random.rand()
        j = np.argwhere(r <= p_cdf)[0]

        Y.append(j)

    return np.array(Y)

def simbased_binomial_sample(N, p, L):
    k = np.random.binomial(N, p, 1)[0]

    return simbased_sample(k, L)
