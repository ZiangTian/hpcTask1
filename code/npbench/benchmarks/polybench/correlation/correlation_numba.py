import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(M, float_n, data):

    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M-1):
        corr[i+1:M, i] = corr[i, i+1:M] = data[:, i] @ data[:, i+1:M]

    return corr


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(M, float_n, data):

    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M-1):
        corr[i+1:M, i] = corr[i, i+1:M] = data[:, i] @ data[:, i+1:M]

    return corr


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_prange(M, float_n, data):

    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in nb.prange(M-1):
        corr[i+1:M, i] = corr[i, i+1:M] = data[:, i] @ data[:, i+1:M]

    return corr


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(M, float_n, data):

    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.sum((data - mean) ** 2, axis=0) / float_n)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M-1):
        corr[i+1:M, i] = corr[i, i+1:M] = data[:, i] @ data[:, i+1:M]

    return corr


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(M, float_n, data):

    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.sum((data - mean) ** 2, axis=0) / float_n)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M-1):
        corr[i+1:M, i] = corr[i, i+1:M] = data[:, i] @ data[:, i+1:M]

    return corr


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(M, float_n, data):

    # mean = np.mean(data, axis=0)
    mean = np.sum(data, axis=0) / float_n
    # stddev = np.std(data, axis=0)
    stddev = np.sqrt(np.sum((data - mean) ** 2, axis=0) / float_n)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in nb.prange(M-1):
        corr[i+1:M, i] = corr[i, i+1:M] = data[:, i] @ data[:, i+1:M]

    return corr
