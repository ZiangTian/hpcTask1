import numpy as np
import dace as dc


N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):
    
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
