import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):

    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):

    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
