import numpy as np
import dace as dc


N, npt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'npt'))


@dc.program
def azimint_naive(data: dc.float64[N], radius: dc.float64[N]):
    # rmax = radius.max()
    rmax = np.amax(radius)
    # res = np.zeros(npt, dtype=np.float64)
    res = np.ndarray(npt, dtype=np.float64)
    res[:] = 0
    # for i in range(npt):
    for i in dc.map[0:npt]:  # Optimization
        r1 = rmax * i / npt
        r2 = rmax * (i+1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        # values_r12 = data[mask_r12]
        # res[i] = np.mean(values_r12)
        on_values = 0
        tmp = np.float64(0)
        # for j in range(N):
        for j in dc.map[0:N]:
            if mask_r12[j]:
                tmp += data[j]
                on_values += 1
        res[i] = tmp / on_values
    return res