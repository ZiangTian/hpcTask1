import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(A, B, C, D):

    return A @ B @ C @ D


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(A, B, C, D):

    return A @ B @ C @ D
