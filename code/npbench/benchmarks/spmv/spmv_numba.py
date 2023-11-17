# Sparse Matrix-Vector Multiplication (SpMV)
import numpy as np
import numba as nb


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in range(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(A_row, A_col, A_val, x):
    y = np.empty(A_row.size - 1, A_val.dtype)

    for i in nb.prange(A_row.size - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y
