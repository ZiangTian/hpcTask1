import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(path):

    for k in range(path.shape[0]):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(path):

    for k in range(path.shape[0]):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(path):

    for k in range(path.shape[0]):
        # path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
        for i in range(path.shape[0]):
            path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(path):

    for k in range(path.shape[0]):
        # path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
        for i in range(path.shape[0]):
            path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])


@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(path):

    for k in range(path.shape[0]):
        # path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
        for i in nb.prange(path.shape[0]):
            path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])
