import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(a):
    trace = 0.0
    for i in nb.prange(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace
