import dace
import numpy
import timeit

from numba import jit, prange

N = 1000000
npt = 1000


def azimint_naive_numpy(data, radius):
    # rmax = radius.max()
    rmax = numpy.amax(radius)
    res = numpy.zeros(npt, dtype=numpy.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i+1) / npt
        mask_r12 = numpy.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
    return res


@jit(nopython=True, parallel=True, fastmath=True)
def azimint_naive_numba(data, radius):
    # rmax = radius.max()
    rmax = numpy.amax(radius)
    res = numpy.zeros(npt, dtype=numpy.float64)
    # for i in prange(npt):
    #     r1 = rmax * i / npt
    #     r2 = rmax * (i+1) / npt
    #     mask_r12 = numpy.logical_and((r1 <= radius), (radius < r2))
    #     # values_r12 = data[mask_r12]
    #     # res[i] = values_r12.mean()
    #     on_values = 0
    #     for j in range(N):
    #         if mask_r12[j]:
    #             res[i] += data[j]
    #             on_values += 1
    #     res[i] /= on_values
    #     # values_r12 = numpy.positive(data, where=mask_r12)
    #     # res[i] = numpy.sum(values_r12) / numpy.sum(mask_r12))
    for i in prange(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i+1) / npt
        on_values = 0
        for j in range(N):
            if r1 <= radius[j] and radius[j] < r2:
                res[i] += data[j]
                on_values += 1
        res[i] /= on_values
    return res


@dace.program
def azimint_naive_dace(data: dace.float64[N], radius: dace.float64[N]):
    # rmax = radius.max()
    rmax = numpy.amax(radius)
    # res = numpy.zeros(npt)
    res = numpy.ndarray(npt, dtype=numpy.float64)
    res[:] = 0
    # mask_r12 = numpy.ndarray(N, dtype=numpy.bool_)
    # for i in dace.map[0:npt]:
    #     r1 = rmax * i / npt
    #     r2 = rmax * (i+1) / npt
    #     mask_r12[:] = numpy.logical_and((r1 <= radius), (radius < r2))
    #     # values_r12 = data[mask_r12]
    #     # res[i] = values_r12.mean()
    #     on_values = 0
    #     for j in range(N):
    #         if mask_r12[j]:
    #             res[i] += data[j]
    #             on_values += 1
    #     res[i] /= on_values
    #     # values_r12 = numpy.positive(data, where=mask_r12)
    #     # res[i] = numpy.sum(values_r12) / numpy.sum(mask_r12)
    for i in dace.map[0:npt]:
        r1 = rmax * i / npt
        r2 = rmax * (i+1) / npt
        on_values = 0
        for j in range(N):
            if r1 <= radius[j] and r2 > radius[j]:
                res[i] += data[j]
                on_values += 1
        res[i] /= on_values
    return res


if __name__ == "__main__":
    data = numpy.random.rand(N).astype(numpy.float64)
    radius = numpy.random.rand(N).astype(numpy.float64)

    dace_exec = azimint_naive_dace.compile()

    np_res = azimint_naive_numpy(data, radius)
    print("Done")
    nb_res = azimint_naive_numba(data, radius)
    print("Done")
    # assert(numpy.allclose(np_res, nb_res))
    dc_res = dace_exec(data=data, radius=radius)
    print("Done")
    # assert(numpy.allclose(np_res, dc_res))
    # assert(numpy.allclose(nb_res, dc_res))

    time = timeit.repeat("azimint_naive_numpy(data, radius)", setup="pass", repeat=10, number=1, globals=globals())
    print("NumPy Median time: {}".format(numpy.median(time)))
    time = timeit.repeat("azimint_naive_numba(data, radius)", setup="pass", repeat=10, number=1, globals=globals())
    print("Numba Median time: {}".format(numpy.median(time)))
    time = timeit.repeat("dace_exec(data=data, radius=radius)", setup="pass", repeat=10, number=1, globals=globals())
    print("DaCe Median time: {}".format(numpy.median(time)))
