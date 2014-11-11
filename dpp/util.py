import numpy as np

def decompose_kernel(L):
    D, V = np.linalg.eigh(L)
    D = np.real(D)
    D[D < 0] = 0
    idx = np.argsort(D)
    D = D[idx]
    V = np.real(V[:, idx])
    return D, V

def esym_poly(k, lam):
  N = lam.size
  E = np.zeros((k+1, N+1))
  E[0, :] = np.ones((1, N+1))
  for l in range(1, k+1):
    for n in range(1, N+1):
      E[l, n] = E[l, n-1] + lam[n-1]*E[l-1, n-1]

  return E

def expected_cardinality(lam):
    return np.sum(lam/(1+lam))

def E_Y(lam):
    return np.sum(lam/(1+lam))