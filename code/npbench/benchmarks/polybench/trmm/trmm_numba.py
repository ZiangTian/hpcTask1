import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(alpha, A, B):

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def nopython_mode_parallel(alpha, A, B):

#     for i in range(B.shape[0]):
#         for j in range(B.shape[1]):
#             B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
#     B *= alpha


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(alpha, A, B):

    for i in range(B.shape[0]):
        for j in nb.prange(B.shape[1]):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
