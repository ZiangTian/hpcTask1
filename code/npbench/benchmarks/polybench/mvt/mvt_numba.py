import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(x1, x2, y_1, y_2, A):

    x1 += A @ y_1
    x2 += y_2 @ A


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(x1, x2, y_1, y_2, A):

    x1 += A @ y_1
    x2 += y_2 @ A
