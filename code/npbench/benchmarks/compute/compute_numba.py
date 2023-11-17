import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(array_1, array_2, a, b, c):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(array_1, array_2, a, b, c):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(array_1, array_2, a, b, c):
    # return np.clip(array_1, 2, 10) * a + array_2 * b + c
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(array_1, array_2, a, b, c):
    # return np.clip(array_1, 2, 10) * a + array_2 * b + c
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c
