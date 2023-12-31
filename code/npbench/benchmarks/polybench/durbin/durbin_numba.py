import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(r):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = - (r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha

    return y


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(r):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = - (r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha

    return y
