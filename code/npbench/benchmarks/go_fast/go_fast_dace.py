import numpy as np
import dace as dc


N = dc.symbol('N', dtype=dc.int64)


@dc.program
def go_fast(a: dc.float64[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace


if __name__ == "__main__":

    from dace.transformation.optimizer import Optimizer
    from dace.transformation.dataflow import MapFusion, Vectorization
    dc_sdfg = go_fast.to_sdfg(strict=False)
    dc_sdfg.apply_strict_transformations()
    dc_sdfg.apply_transformations_repeated([MapFusion])

    matches = []
    for xform in Optimizer(dc_sdfg).get_pattern_matches(patterns=[Vectorization]):
        print('Match:', xform.print_match(dc_sdfg))
        matches.append(xform)
    for xform in matches:
        xform.apply(dc_sdfg)
    dc_sdfg.compile()
