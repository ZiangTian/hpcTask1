import timeit
import numpy as orig_np
import legate.numpy as np


def azimint_naive(data, radius, npt):
    rmax = radius.max()
    res = np.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i+1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
        # on_values = 0
        # for j in range(len(data)):
        #     if r1 <= radius[j] and radius[j] < r2:
        #         res[i] += data[j]
        #         on_values += 1
        # res[i] /= on_values
    return res


def initialize(N):
    data = orig_np.random.rand(N).astype(np.float64)
    radius = orig_np.random.rand(N).astype(np.float64)
    return data, radius


if __name__ == "__main__":

    # Initialization
    N = 1000000
    npt = 1000
    orig_data, orig_radius = initialize(N)
    data = np.empty_like(orig_data)
    data[:] = orig_data
    radius = np.empty_like(orig_radius)
    radius[:] = orig_radius
    np_args = (data, radius, npt)

    # First execution
    azimint_naive(data, radius, npt)

    # Benchmark
    time = timeit.repeat("azimint_naive(data, radius, npt)",
                         setup="pass", repeat=10, number=1, globals=globals())
    print("Legate Median time: {}".format(np.median(time)))
