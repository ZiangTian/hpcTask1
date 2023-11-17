import argparse
import itertools
import os
import numpy as np
from npbench import str2bool, benchmark, validation


def initialize(N):
    data = np.random.rand(N).astype(np.float64)
    radius = np.random.rand(N).astype(np.float64)
    return data, radius


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--module-name", type=str, nargs="?", default="azimint_naive_numba")
    parser.add_argument("--module-desc", type=str, nargs="?", default="Numba")
    parser.add_argument("--func-name", type=str, nargs="?", default="without_prange")
    parser.add_argument("--func-desc", type=str, nargs="?", default="without prange annotation")
    parser.add_argument("--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("--repeat", type=int, nargs="?", default=10)
    parser.add_argument("--papi", type=str2bool, nargs="?", default=True)
    args = vars(parser.parse_args())

    # Import NumPy implementation
    try:
        from azimint_naive_numpy import azimint_naive as np_impl
    except Exception as e:
        raise e
        # raise FileNotFoundError("Failed to load the NumPy implementation")

    # Import contender implementation
    try:
        exec("from {m} import {f} as ct_impl".format(
            m=args["module_name"], f=args["func_name"]))
    except Exception as e:
        raise e
        # raise FileNotFoundError("Failed to load the contender implementation")

    # Initialization
    N = 1000000
    npt = 1000
    data, radius = initialize(N)

    for nopython, parallel, fastmath in itertools.product(
            [False, True], [False, True], [False, True]):
        
        os.environ['OMP_NUM_THREADS'] = '1'
        
        post_str = "nopython={n}, parallel={p}, fastmath={f}".format(
                    n=nopython, p=parallel, f=fastmath)

        # First execution
        exec_str = "ct_impl(data, radius, npt, {})".format(post_str)
        setup_str = "pass"
        try:
            report_str = "{m} {f} ({p}) first execution".format(
                m=args["module_desc"], f=args["func_desc"], p=post_str)
            ct_out, fe_time = benchmark(exec_str, setup=setup_str, out_text=report_str, context=globals())
        except Exception as e:
            raise e

        # Generate reference output from NumPy
        if args["validate"]:
            report_str = "{m} {f} ({p})".format(
                m=args["module_desc"], f=args["func_desc"], p=post_str)
            np_out = np_impl(data, radius, npt)
            validation(np_out, ct_out, report_str)
        
        # Benchmark
        try:
            report_str = "{m} {f} ({p}) median execution".format(
                m=args["module_desc"], f=args["func_desc"], p=post_str)
            repeat_num = args["repeat"]
            benchmark(exec_str, setup=setup_str, out_text=report_str, repeat=repeat_num, context=globals())
        except Exception as e:
            raise e

        # PAPI profiling
        if args["papi"]:
            try:
                # Set OMP_NUM_THREADS to 1
                # os.environ['OMP_NUM_THREADS'] = '1'

                from pypapi import papi_high
                from pypapi import events as papi_events

                # events = [papi_events.PAPI_DP_OPS]  #, papi_events.PAPI_L1_TCM, papi_events.PAPI_L2_TCM]
                events = [papi_events.PAPI_L2_TCM, papi_events.PAPI_L2_TCA]
                        #   papi_events.PAPI_L3_TCM, papi_events.PAPI_L3_TCA]
                
                if setup_str != "pass":
                    eval(setup_str, globals())
                papi_high.start_counters(events)
                papi_high.read_counters()
                eval(exec_str, globals())
                counters = papi_high.read_counters()
                papi_high.stop_counters()
                # assert(counters[1] >= counters[0])
                print(100*counters[0]/counters[1])
            except Exception as e:
                raise e
