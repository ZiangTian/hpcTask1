import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(L, x, b):

    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(L, x, b):

    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
