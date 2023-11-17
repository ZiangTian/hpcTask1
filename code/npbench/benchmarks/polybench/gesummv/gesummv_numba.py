import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x
