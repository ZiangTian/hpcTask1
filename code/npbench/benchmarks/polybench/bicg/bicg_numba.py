import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(A, p, r):

    return r @ A, A @ p


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(A, p, r):

    return r @ A, A @ p
