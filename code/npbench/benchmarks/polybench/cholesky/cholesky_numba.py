import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(A):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def nopython_mode_parallel(A):

#     A[0, 0] = np.sqrt(A[0, 0])
#     for i in range(1, A.shape[0]):
#         for j in range(i):
#             A[i, j] -= np.dot(A[i, :j], A[j, :j])
#             A[i, j] /= A[j, j]
#         A[i, i] -= np.dot(A[i, :i], A[i, :i])
#         A[i, i] = np.sqrt(A[i, i])


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(A):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in nb.prange(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


@nb.jit(nopython=True)
def kernel2(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
