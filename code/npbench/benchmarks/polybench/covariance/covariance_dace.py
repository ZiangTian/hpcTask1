import numpy as np
import dace as dc


M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(float_n: dc.float64, data: dc.float64[N, M]):

    mean = np.mean(data, axis=0)
    # data -= mean
    np.subtract(data, mean, out=data)
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
        cov[i:M, i] = cov[i, i:M]

    return cov
