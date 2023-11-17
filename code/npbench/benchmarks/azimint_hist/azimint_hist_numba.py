import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    histw = np.histogram(radius, npt, weights=data)[0]
    return histw / histu


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    histw = np.histogram(radius, npt, weights=data)[0]
    return histw / histu


@nb.jit(nopython=True, parallel=False, fastmath=True)
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@nb.jit(nopython=True, parallel=True, fastmath=True)
def get_bin_edges_parallel(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@nb.jit(nopython=True, parallel=True, fastmath=True)
def get_bin_edges_prange(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in nb.prange(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@nb.jit(nopython=True, fastmath=True)
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    return int(n * (x - a_min) / (a_max - a_min))
    # bin = int(n * (x - a_min) / (a_max - a_min))

    # if bin < 0 or bin >= n:
    #     return None
    # else:
    #     return bin


@nb.jit(nopython=True, parallel=False, fastmath=True)
def histogram(a, bins, weights):
    hist = np.zeros((bins,), dtype=a.dtype)
    bin_edges = get_bin_edges(a, bins)

    for i in range(a.shape[0]):
        bin = compute_bin(a[i], bin_edges)
        # if bin is not None:
        #     hist[int(bin)] += weights[i]
        hist[bin] += weights[i]

    return hist, bin_edges


@nb.jit(nopython=True, parallel=True, fastmath=True)
def histogram_parallel(a, bins, weights):
    hist = np.zeros((bins,), dtype=a.dtype)
    bin_edges = get_bin_edges_parallel(a, bins)

    for i in range(a.shape[0]):
        bin = compute_bin(a[i], bin_edges)
        # if bin is not None:
        #     hist[int(bin)] += weights[i]
        hist[bin] += weights[i]

    return hist, bin_edges


@nb.jit(nopython=True, parallel=True, fastmath=True)
def histogram_prange(a, bins, weights):
    hist = np.zeros((bins,), dtype=a.dtype)
    bin_edges = get_bin_edges_parallel(a, bins)

    for i in nb.prange(a.shape[0]):
        bin = compute_bin(a[i], bin_edges)
        # if bin is not None:
        #     hist[int(bin)] += weights[i]
        hist[bin] += weights[i]

    return hist, bin_edges


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    # histw = np.histogram(radius, npt, weights=data)[0]
    histw = histogram(radius, npt, weights=data)[0]
    return histw / histu


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    # histw = np.histogram(radius, npt, weights=data)[0]
    histw = histogram_parallel(radius, npt, weights=data)[0]
    return histw / histu


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    # histw = np.histogram(radius, npt, weights=data)[0]
    histw = histogram_prange(radius, npt, weights=data)[0]
    return histw / histu
