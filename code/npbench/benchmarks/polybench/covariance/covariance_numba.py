import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(M, float_n, data):

    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in range(M):
    #     for j in range(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(M, float_n, data):

    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in range(M):
    #     for j in range(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_prange(M, float_n, data):

    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in nb.prange(M):
    #     for j in nb.prange(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in mb.prange(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(M, float_n, data):

    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in range(M):
    #     for j in range(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(M, float_n, data):

    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in range(M):
    #     for j in range(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(M, float_n, data):

    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in nb.prange(M):
    #     for j in nb.prange(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov
