import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)