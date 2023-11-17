import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(alpha, beta, C, A, B):

    temp2 = np.empty((C.shape[1], ), dtype=C.dtype)
    C *= beta
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def nopython_mode_parallel(alpha, beta, C, A, B):

#     temp2 = np.empty((C.shape[1], ), dtype=C.dtype)
#     C *= beta
#     for i in range(C.shape[0]):
#         for j in range(C.shape[1]):
#             C[:i, j] += alpha * B[i, j] * A[i, :i]
#             temp2[j] = B[:i, j] @ A[i, :i]
#         C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(alpha, beta, C, A, B):

    temp2 = np.empty((C.shape[1], ), dtype=C.dtype)
    C *= beta
    for i in range(C.shape[0]):
        for j in nb.prange(C.shape[1]):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
