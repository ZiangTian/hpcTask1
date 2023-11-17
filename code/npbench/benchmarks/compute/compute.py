import argparse
import pathlib
import numpy as np
from npbench import run, run_dace, run_numba, str2bool


# Module name
module_name = "compute"
func_name = "compute"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "kernels",
    dwarf = "monte_carlo",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "array_1, array_2, a, b, c",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "array_1, array_2, a, b, c",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "array_1, array_2, a, b, c",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "garray_1, garray_2, a, b, c",
        setup_str = "garray_1, garray_2 = cp.asarray(array_1), cp.asarray(array_2)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "array_1=array_1, array_2=array_2, a=a, b=b, c=c, M=M, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "array_1=garray_1, array_2=garray_2, a=a, b=b, c=c, M=M, N=N",
        setup_str = "garray_1, garray_2 = cp.asarray(array_1), cp.asarray(array_2)",
        report_str = "DaCe GPU"
    )
)

# # Argument strings
# numpy_arg_str = "array_1, array_2, a, b, c"
# cython_arg_str = "array_1, array_2, a, b, c"
# numba_arg_str = "array_1, array_2, a, b, c"
# dace_arg_str = "array_1=array_1, array_2=array_2, a=a, b=b, c=c, M=M, N=N"
# cupy_arg_str = "gpu_array_1, gpu_array_2, a, b, c"

# # Setup string
# numpy_setup_str = "pass"
# cython_setup_str = "pass"
# numba_setup_str = "pass"
# dace_setup_str = "pass"

# # Report string
# numpy_report_str = "NumPy"
# cython_report_str = "Cython"
# numba_report_str = "Numba"
# dace_report_str = "DaCe"
# cupy_report_str = "CuPy"


def initialize(M, N):
    array_1 = np.random.uniform(0, 1000, size=(M, N)).astype(np.int32)
    array_2 = np.random.uniform(0, 1000, size=(M, N)).astype(np.int32)
    a = np.intc(4)
    b = np.intc(3)
    c = np.intc(9)
    return array_1, array_2, a, b, c


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    M = 50000
    N = 50000
    array_1, array_2, a, b, c = initialize(M, N)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())

    # # GPU
    # if args["framework"] == "cupy":
    #     import cupy as cp
    #     gpu_data, gpu_radis = cp.asarray(data), cp.asarray(radius)

    # if args["validate"]:
    #     if args["framework"] == "numpy":
    #         args["validate"] = False
    #         np_out = None
    #     else:
    #         # Import NumPy implementation and get reference output
    #         try:
    #             from compute_numpy import compute as np_impl
    #             if numpy_setup_str and numpy_setup_str != "pass":
    #                 exec(numpy_setup_str)
    #             # np_out = np_impl(*np_args)
    #             np_out = eval("np_impl({})".format(numpy_arg_str), globals())
    #         except Exception as e:
    #             args["validate"] = False
    #             np_out = None
    #             print("Failed to load the NumPy implementation. Validation is not possible.")
    #             print(e)

    # # Run benchmark
    # if args["framework"] == "numba":
    #     run_numba(module_name, numba_arg_str, numba_setup_str, np_out,
    #               args["mode"], args["validate"], args["repeat"], args["append"], globals())
    # elif args["framework"] == "dace":
    #     run_dace(module_name, func_name, dace_arg_str, dace_setup_str, np_out,
    #              "cpu", args["mode"], args["validate"], args["repeat"], args["append"], globals())
    # else:
    #     # Import contender implementation
    #     try:
    #         exec("from {m}_{fw} import {f} as ct_impl".format(
    #             m=module_name, fw=args["framework"], f=func_name))
    #     except Exception as e:
    #         print("Failed to load the {} implementation.".format(args["framework"]))
    #         raise(e)

    #     arg_str = eval("{}_arg_str".format(args["framework"]))       
    #     exec_str = "ct_impl({})".format(arg_str)
    #     setup_str = eval("{}_setup_str".format(args["framework"]))
    #     report_str = eval("{}_report_str".format(args["framework"]))

    #     run(exec_str, setup_str, report_str, np_out, args["mode"],
    #         args["validate"], args["repeat"], args["append"], globals())

    # # # First execution
    # # npX = compute_numpy.compute(array_1, array_2, a, b, c)
    # # # cyX = compute_cython_1.compute(array_1, array_2, a, b, c)
    # # nbX = compute_numba.compute(array_1, array_2, a, b, c)
    # # dace_exec = compute_dace.compute.compile()
    # # dcX = dace_exec(array_1=array_1, array_2=array_2, a=a, b=b, c=c, M=M, N=N)
    # # # lgX = compute_legate.compute(array_1, array_2, a, b, c)

    # # # Benchmark
    # # time = timeit.repeat("compute_numpy.compute(array_1, array_2, a, b, c)",
    # #                      setup="pass", repeat=20, number=1, globals=globals())
    # # print("NumPy median time: {}".format(np.median(time)))
    # # # time = timeit.repeat("compute_cython_1.compute(array_1, array_2, a, b, c)",
    # # #                      setup="pass", repeat=20, number=1, globals=globals())
    # # print("Cython median time: {}".format(np.median(time)))
    # # time = timeit.repeat("compute_numba.compute(array_1, array_2, a, b, c)",
    # #                      setup="pass", repeat=20, number=1, globals=globals())
    # # print("Numba median time: {}".format(np.median(time)))
    # # time = timeit.repeat(
    # #     "dace_exec(array_1=array_1, array_2=array_2, a=a, b=b, c=c, M=M, N=N)",
    # #     setup="pass", repeat=20, number=1, globals=globals())
    # # print("DaCe median time: {}".format(np.median(time)))
    # # # time = timeit.repeat("compute_legate.compute(array_1, array_2, a, b, c)",
    # # #                      setup="pass", repeat=20, number=1, globals=globals())
    # # # print("Legate median time: {}".format(np.median(time)))

    # # # Validation
    # # assert(np.allclose(npX, cyX))
    # # assert(np.allclose(npX, nbX))
    # # assert(np.allclose(npX, dcX))
    # # # assert(np.allclose(npX, lgX))
