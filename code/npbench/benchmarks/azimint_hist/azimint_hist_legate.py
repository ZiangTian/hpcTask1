# import timeit
# import legate.numpy as np

import argparse
import legate.numpy as np
from npbench import run, run_dace, run_numba, str2bool


# Module name
module_name = "azimint_hist"
func_name = "azimint_hist"

# Argument strings
numpy_arg_str = "data, radius, npt"
legate_arg_str = "data, radius, npt"

# Setup string
numpy_setup_str = "pass"
legate_setup_str = "pass"

# Report string
numpy_report_str = "NumPy"
legate_report_str = "Legate"


def azimint_hist(data, radius, npt):
    histu = np.histogram(radius, npt)[0]
    histw = np.histogram(radius, npt, weights=data)[0]
    # return histw / histu
    res = histw / histu
    assert(res[0])
    return res


def initialize(N):
    data = np.random.rand(N).astype(np.float64)
    radius = np.random.rand(N).astype(np.float64)
    return data, radius


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="legate")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="median")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=True)
    args = vars(parser.parse_args())

    # Initialization
    N = 1000000
    npt = 1000
    data, radius = initialize(N)
    np_args = (data, radius, npt)

    # GPU
    if args["framework"] == "cupy":
        import cupy as cp
        gpu_data, gpu_radis = cp.asarray(data), cp.asarray(radius)

    if args["validate"]:
        if args["framework"] == "numpy":
            args["validate"] = False
            np_out = None
        else:
            # Import NumPy implementation and get reference output
            try:
                from azimint_hist_numpy import azimint_hist as np_impl
                np_out = np_impl(*np_args)
            except Exception as e:
                args["validate"] = False
                np_out = None
                print("Failed to load the NumPy implementation. Validation is not possible.")
                print(e)

    # Run benchmark
    if args["framework"] == "numba":
        run_numba(module_name, numba_arg_str, numba_setup_str, np_out,
                  args["mode"], args["validate"], args["repeat"], args["append"], globals())
    elif args["framework"] == "dace":
        run_dace(module_name, func_name, dace_arg_str, dace_setup_str, np_out,
                 "cpu", args["mode"], args["validate"], args["repeat"], args["append"], globals())
    elif args["framework"] == "legate":
        arg_str = eval("{}_arg_str".format(args["framework"]))       
        exec_str = "{f}({a})".format(f=func_name, a=arg_str)
        setup_str = eval("{}_setup_str".format(args["framework"]))
        report_str = eval("{}_report_str".format(args["framework"]))

        run(exec_str, setup_str, report_str, np_out, args["mode"],
            args["validate"], args["repeat"], args["append"], globals())
    else:
        # Import contender implementation
        try:
            exec("from {m}_{fw} import {f} as ct_impl".format(
                m=module_name, fw=args["framework"], f=func_name))
        except Exception as e:
            print("Failed to load the {} implementation.".format(args["framework"]))
            raise(e)

        arg_str = eval("{}_arg_str".format(args["framework"]))       
        exec_str = "ct_impl({})".format(arg_str)
        setup_str = eval("{}_setup_str".format(args["framework"]))
        report_str = eval("{}_report_str".format(args["framework"]))

        run(exec_str, setup_str, report_str, np_out, args["mode"],
            args["validate"], args["repeat"], args["append"], globals())
    
    # # Initialization
    # N = 1000000
    # npt = 1000
    # data = np.random.rand(N).astype(np.float64)
    # radius = np.random.rand(N).astype(np.float64)

    # # First execution
    # azimint_hist(data, radius, npt)

    # # Benchmark
    # time = timeit.repeat("azimint_hist(data, radius, npt)",
    #                      setup="pass", repeat=10, number=1, globals=globals())
    # print("Legate Median time: {}".format(np.median(time)))
