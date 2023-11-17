import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(alpha, beta, A, B, C, D):

    D[:] = alpha * A @ B @ C + beta * D


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(alpha, beta, A, B, C, D):

    D[:] = alpha * A @ B @ C + beta * D
