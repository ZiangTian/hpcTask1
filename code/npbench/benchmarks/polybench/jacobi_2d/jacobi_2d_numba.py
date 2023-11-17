import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(TSTEPS, A, B):
    
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] +
                                 A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] +
                                 B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(TSTEPS, A, B):
    
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] +
                                 A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] +
                                 B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])
