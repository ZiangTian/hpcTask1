# cython: language_level=3
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp


import numpy as np
cimport cython
from cython.parallel import prange


# def kernel(M, float_n, data):

#     # mean = np.sum(data, axis=0) / float_n
#     mean = np.mean(data, axis=0)
#     # stddev = np.sqrt(np.sum((data - mean) ** 2, axis=0) / float_n)
#     stddev = np.std(data, axis=0)
#     stddev[stddev <= 0.1] = 1.0
#     data -= mean
#     data /= np.sqrt(float_n) * stddev
#     corr = np.zeros((M, M), dtype=data.dtype)
#     for i in range(M-1):
#     # cdef Py_ssize_t mm1 = data.shape[1]-1
#     # for i in prange(mm1, nogil=True):
#         for j in range(i+1, M):
#             corr[i, j] = np.sum(data[:, i] * data[:, j])
#             corr[j, i] = corr[i, j]
#     np.fill_diagonal(corr, 1.0)

#     return corr


ctypedef fused my_type:
    double


@cython.boundscheck(False)
@cython.wraparound(False)
def kernel(int M, my_type float_n, my_type[:, ::1] data):

    if my_type is double:
        dtype = np.double

    # mean = np.sum(data, axis=0) / float_n
    mean = np.mean(data, axis=0)
    # stddev = np.sqrt(np.sum((data - mean) ** 2, axis=0) / float_n)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev

    corr = np.zeros((M, M), dtype=dtype)
    cdef my_type[:, ::1] corr_view = corr

    cdef my_type tmp
    cdef Py_ssize_t i, j, k

    # # for i in prange(M-1, nogil=True):
    # for i in range(M-1):
    #     for j in range(i+1, M):
    #         # for k in range(data.shape[0]):
    #         for k in prange(data.shape[0], nogil=True):
    #             corr_view[i, j] += data[k, i] * data[k, j]
    #         corr_view[j, i] = corr_view[i, j]
    #         # corr[i, j] = np.sum(data[:, i] * data[:, j])
    #         # corr_view[i, j] = np.dot(data[:, i], data[:, j])
    #         # corr_view[j, i] = corr_view[i, j]
    for i in range(M-1):
        corr[i+1:M, i] = corr[i, i+1:M] = np.transpose(data[:, i+1:M]) @ data[:, i]
    np.fill_diagonal(corr, 1.0)

    return corr
