import numpy as np

def standard_sample(lam, V_full):
  D = lam/(1+lam)
  v = np.random.rand(1, len(lam)) <= D
  k = np.sum(v)
  V = V_full[:, v.flatten()]
  Y = np.zeros((k, 1))

  for i in range(k-1, -1, -1):
    # Sample
    Pr = np.sum(V**2, axis=1)
    Pr = Pr/sum(Pr)
    C = np.cumsum(Pr)
    jj = np.argwhere(np.random.rand() <= C)[0]
    Y[i] = jj

    # Update V
    if i > 0:
      j = np.argwhere(V[int(Y[i]), :])[0]
      Vj = V[:, j]
      V = np.delete(V, j, 1)
      V = V - np.outer(Vj, V[int(Y[i]), :]/Vj[int(Y[i])])

      # QR decomposition, which is more numerically stable (and faster) than Gram
      # Schmidt
      V, r = np.linalg.qr(V)

  return Y


