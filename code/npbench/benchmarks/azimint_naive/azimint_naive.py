import argparse
import pathlib
import numpy as np
from npbench import run, run_dace, run_numba, str2bool


# Module name
module_name = "azimint_naive"
func_name = "azimint_naive"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "crystallography",
    dwarf = "spectral_methods",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "data, radius, npt",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "data, radius, npt",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "data, radius, npt",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gdata, gradius, npt",
        setup_str = "gdata, gradius = cp.asarray(data), cp.asarray(radius)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "data=data, radius=radius, npt=npt, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "data=gdata, radius=gradius, npt=npt, N=N",
        setup_str = "gdata, gradius = cp.asarray(data), cp.asarray(radius)",
        report_str = "DaCe GPU"
    )
)

# # Argument strings
# numpy_arg_str = "data, radius, npt"
# numba_arg_str = "data, radius, npt"
# dace_arg_str = "data=data, radius=radius, npt=npt, N=N"
# cupy_arg_str = "gpu_data, gpu_radius, npt"

# # Setup string
# numpy_setup_str = "pass"
# numba_setup_str = "pass"
# dace_setup_str = "pass"

# # Report string
# numpy_report_str = "NumPy"
# numba_report_str = "Numba"
# dace_report_str = "DaCe"
# cupy_report_str = "CuPy"


def initialize(N):
    data = np.random.rand(N).astype(np.float64)
    radius = np.random.rand(N).astype(np.float64)
    return data, radius


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="pythran")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    # parser.add_argument("--module-name", type=str, nargs="?", default="azimint_naive_numpy")
    # parser.add_argument("--module-desc", type=str, nargs="?", default="NumPy")
    # parser.add_argument("--func-name", type=str, nargs="?", default="azimint_naive")
    # parser.add_argument("--func-desc", type=str, nargs="?", default="original code")
    args = vars(parser.parse_args())

    # Initialization
    N = 1000000
    npt = 1000
    data, radius = initialize(N)
    np_args = (data, radius, npt)

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
    #             from azimint_naive_numpy import azimint_naive as np_impl
    #             np_out = np_impl(*np_args)
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
    # # exec_str = "ct_impl(data, radius, npt)"
    # # setup_str = "pass"
    # # try:
    # #     report_str = "{m} {f} first execution".format(
    # #         m=args["module_desc"], f=args["func_desc"])
    # #     ct_out, fe_time = benchmark(exec_str, setup=setup_str, out_text=report_str, context=globals())
    # # except Exception as e:
    # #     raise e

    # # # Generate reference output from NumPy
    # # if args["validate"]:
    # #     np_out = np_impl(data, radius, npt)
    # #     validation(np_out, ct_out)
    
    # # # Benchmark
    # # try:
    # #     report_str = "{m} {f} median execution".format(
    # #         m=args["module_desc"], f=args["func_desc"])
    # #     repeat_num = args["repeat"]
    # #     benchmark(exec_str, setup=setup_str, out_text=report_str, repeat=repeat_num, context=globals())
    # # except Exception as e:
    # #     raise e

    # # # PAPI profiling
    # # if args["papi"]:
    # #     try:
    # #         from pypapi import papi_high
    # #         from pypapi import events as papi_events

    # #         events = [papi_events.PAPI_DP_OPS]  #, papi_events.PAPI_L1_TCM, papi_events.PAPI_L2_TCM]
            
    # #         if setup_str != "pass":
    # #             eval(setup_str, globals())
    # #         papi_high.start_counters(events)
    # #         eval(exec_str, globals())
    # #         counters = papi_high.read_counters()
    # #         papi_high.stop_counters()
    # #         print(counters)
    # #     except Exception as e:
    # #         raise e

    # # # First execution
    # # np_res = azimint_naive_numpy.azimint_naive(data, radius, npt)
    # # nb_res = azimint_naive_numba.azimint_naive(data, radius, npt)
    # # dace_exec = azimint_naive_dace.azimint_naive.compile()
    # # dc_res= dace_exec(data=data, radius=radius, npt=npt, N=N)

    # # # Benchmark
    # # time = timeit.repeat("azimint_naive_numpy.azimint_naive(data, radius, npt)",
    # #                      setup="pass", repeat=10, number=1, globals=globals())
    # # print("NumPy Median time: {}".format(np.median(time)))
    # # time = timeit.repeat("azimint_naive_numba.azimint_naive(data, radius, npt)",
    # #                      setup="pass", repeat=10, number=1, globals=globals())
    # # print("Numba Median time: {}".format(np.median(time)))
    # # time = timeit.repeat("dace_exec(data=data, radius=radius, npt=npt, N=N)",
    # #                      setup="pass", repeat=10, number=1, globals=globals())
    # # print("DaCe Median time: {}".format(np.median(time)))

    # # # Validation
    # # if not np.allclose(np_res, nb_res):
    # #     print("Numba relerror: {}".format(np.linalg.norm(np_res - nb_res)/np.linalg.norm(np_res)))
    # # if not np.allclose(np_res, dc_res):
    # #     print("DaCe relerror: {}".format(np.linalg.norm(np_res - dc_res)/np.linalg.norm(np_res)))
