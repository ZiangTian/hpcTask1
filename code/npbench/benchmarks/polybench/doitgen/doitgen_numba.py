import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(NR, NQ, NP, A, C4):

    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(NR, NQ, NP, A, C4):

    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(NR, NQ, NP, A, C4):

    for r in range(NR):
        for q in range(NQ):
            tmp = A[r, q, :] @ C4
            A[r, q, :] = tmp
    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def nopython_mode_parallel(NR, NQ, NP, A, C4):

#     for r in range(NR):
#         for q in range(NQ):
#             tmp = A[r, q, :] @ C4
#             A[r, q, :] = tmp
#     # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def nopython_mode_prange(NR, NQ, NP, A, C4):

#     for r in nb.prange(NR):
#         for q in nb.prange(NQ):
#             tmp = A[r, q, :] @ C4
#             A[r, q, :] = tmp
#     # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
