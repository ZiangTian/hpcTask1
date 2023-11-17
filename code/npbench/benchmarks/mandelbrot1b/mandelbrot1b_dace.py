import numpy as np
import dace as dc

XN, YN, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'N'])


@dc.program
def linspace(start: dc.float64, stop: dc.float64, X: dc.float64[N]):
    dist = (stop - start) / (N - 1)
    for i in dace.map[0:N]:
        X[i] = start + i * dist


@dc.program
def mandelbrot(xmin: dc.float64, xmax: dc.float64, ymin: dc.float64,
               ymax:dc.float64, maxiter: dc.int64, horizon: dc.float64):
    X = np.ndarray((XN, ), dtype=np.float64)
    Y = np.ndarray((YN, ), dtype=np.float64)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    # C = X + np.reshape(Y, (YN, 1)) * 1j
    C = np.ndarray((YN, XN), dtype=np.complex128)
    for i, j in dc.map[0:YN, 0:XN]:
        C[i, j] = X[j] + Y[i] * 1j
    N = np.ndarray((YN, XN), dtype=np.int64)
    N[:] = 0
    Z = np.ndarray((YN, XN), dtype=np.complex128)
    Z[:] = 0
    for n in range(maxiter):
        I = np.less(np.absolute(Z), horizon)
        # N[I] = n
        N[:] = np.int64(I) * n + np.int64(~I) * N
        # N[:] = I * n + (~I) * N
        Z[:] = np.int64(I) * (Z**2 + C) + np.int64(~I) * Z
        # Z[:] = I * (Z**2 + C) + (~I) * Z
    # N[N == maxiter-1] = 0
    I[:] = (N != maxiter - 1)
    N[:] = np.int64(I) * N
    # N[:] = I * N
    return Z, N
