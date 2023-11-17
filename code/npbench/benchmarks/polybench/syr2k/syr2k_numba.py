import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(alpha, beta, C, A, B):

    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta 
        for k in range(A.shape[1]):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] +
                             B[:i + 1, k] * alpha * A[i, k])


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def nopython_mode_parallel(alpha, beta, C, A, B):

#     for i in range(A.shape[0]):
#         C[i, :i + 1] *= beta 
#         for k in range(A.shape[1]):
#             C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] +
#                              B[:i + 1, k] * alpha * A[i, k])
