import numpy as np
import dace as dc


N, bins, npt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'bins', 'npt'))


# @dc.program
# def get_bin_edges(a: dc.float64[N], bins: dc.int64):
#     bin_edges = np.ndarray((bins+1,), dtype=np.float64)
#     a_min = np.amin(a)
#     a_max = np.amax(a)
#     delta = (a_max - a_min) / bins
#     for i in dc.map[0:bins]:
#         bin_edges[i] = a_min + i * delta

#     bin_edges[bins] = a_max  # Avoid roundoff error on last point
#     return bin_edges


@dc.program
def get_bin_edges(a: dc.float64[N], bin_edges: dc.float64[bins+1]):
    # bin_edges = np.ndarray((bins+1,), dtype=np.float64)
    a_min = np.amin(a)
    a_max = np.amax(a)
    delta = (a_max - a_min) / bins
    for i in dc.map[0:bins]:
        bin_edges[i] = a_min + i * delta

    bin_edges[bins] = a_max  # Avoid roundoff error on last point
    # return bin_edges


@dc.program
def compute_bin(x: dc.float64, bin_edges: dc.float64[bins+1]):
    # assuming uniform bins for now
    a_min = bin_edges[0]
    a_max = bin_edges[bins]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
    # if x == bin_edges[bins]:
        return bins # a_max always in last bin

    return dc.int64(bins * (x - a_min) / (a_max - a_min))


# @dc.program
# def histogram(a: dc.float64[N], bin_edges: dc.float64[bins+1],
#               weights: dc.float64[N]):
#     hist = np.ndarray((bins+1,), dtype=np.float64)
#     # bin_edges = get_bin_edges(a, bins=bins)
#     get_bin_edges(a, bin_edges)

#     if weights is None:
#         for i in dc.map[0:N]:
#             bin = compute_bin(a[i], bin_edges)
#             hist[bin] += 1
#     else:
#         for i in dc.map[0:N]:
#             bin = compute_bin(a[i], bin_edges)
#             hist[bin] += weights[i]

#     return hist


@dc.program
def histogram(a: dc.float64[N], bin_edges: dc.float64[bins+1]):
    hist = np.ndarray((bins,), dtype=np.float64)
    hist[:] = 0
    # bin_edges = get_bin_edges(a)
    get_bin_edges(a, bin_edges)

    for i in dc.map[0:N]:
    # for i in range(N):
        bin = compute_bin(a[i], bin_edges)
        hist[bin] += 1

    return hist


@dc.program
def histogram_weights(a: dc.float64[N], bin_edges: dc.float64[bins+1],
                      weights: dc.float64[N]):
    hist = np.ndarray((bins,), dtype=np.float64)
    hist[:] = 0
    # bin_edges = get_bin_edges(a)
    get_bin_edges(a, bin_edges)

    for i in dc.map[0:N]:
    # for i in range(N):
        bin = compute_bin(a[i], bin_edges)
        hist[bin] += weights[i]

    return hist


@dc.program
def azimint_hist(data: dc.float64[N], radius: dc.float64[N]):
    # histu = np.histogram(radius, npt)[0]
    bin_edges_u = np.ndarray((npt+1,), dtype=np.float64)
    histu = histogram(radius, bin_edges_u)
    # histw = np.histogram(radius, npt, weights=data)[0]
    bin_edges_w = np.ndarray((npt+1,), dtype=np.float64)
    histw = histogram_weights(radius, bin_edges_w, data)
    return histw / histu


if __name__ == "__main__":
    azimint_hist.compile()
