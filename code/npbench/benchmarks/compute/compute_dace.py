import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))

@dc.program
def compute(array_1: dc.int32[M, N], array_2: dc.int32[M, N],
            a: dc.int32, b: dc.int32, c: dc.int32):
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c
