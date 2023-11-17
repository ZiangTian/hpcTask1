import numpy as np


def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    return int(n * (x - a_min) / (a_max - a_min))


def histogram(a, bins):
    hist = np.zeros((bins,), dtype=a.dtype)
    bin_edges = get_bin_edges(a, bins)

    for i in range(a.shape[0]):
        bin = compute_bin(a[i], bin_edges)
        hist[bin] += 1

    return hist, bin_edges


def histogram_w(a, bins, weights):
    hist = np.zeros((bins,), dtype=a.dtype)
    bin_edges = get_bin_edges(a, bins)

    for i in range(a.shape[0]):
        bin = compute_bin(a[i], bin_edges)
        hist[bin] += weights[i]

    return hist, bin_edges


# pythran export azimint_hist(float64[], float64[], int64)
def azimint_hist(data, radius, npt):
    # histu = np.histogram(radius, npt)[0]
    histu = histogram(radius, npt)[0]
    # histw = np.histogram(radius, npt, weights=data)[0]
    histw = histogram_w(radius, npt, weights=data)[0]
    return histw / histu
