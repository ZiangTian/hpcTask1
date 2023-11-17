import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C 


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C 
