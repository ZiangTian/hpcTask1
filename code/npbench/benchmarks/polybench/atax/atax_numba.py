import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(A, x):

    return (A @ x) @ A


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(A, x):

    return (A @ x) @ A
