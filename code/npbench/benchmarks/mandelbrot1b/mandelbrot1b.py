import argparse
import numpy as np
from npbench import run, run_dace, run_numba, str2bool


# Module name
module_name = "mandelbrot1b"
func_name = "mandelbrot"

# Argument strings
numpy_arg_str = "xmin, xmax, ymin, ymax, xn, yn, maxiter"
numba_arg_str = "xmin, xmax, ymin, ymax, xn, yn, maxiter"
dace_arg_str = "xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, maxiter=maxiter, horizon=2.0, XN=xn, YN=yn"
cupy_arg_str = "xmin, xmax, ymin, ymax, xn, yn, maxiter"

# Setup string
numpy_setup_str = "pass"
numba_setup_str = "pass"
dace_setup_str = "pass"

# Report string
numpy_report_str = "NumPy"
numba_report_str = "Numba"
dace_report_str = "DaCe"
cupy_report_str = "CuPy"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="median")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=True)
    args = vars(parser.parse_args())

    # Initialization
    xmin, xmax, xn = -2.25, +0.75, int(3000/3)
    ymin, ymax, yn = -1.25, +1.25, int(3000/3)
    maxiter = 200

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
                from mandelbrot1b_numpy import mandelbrot as np_impl
                if numpy_setup_str and numpy_setup_str != "pass":
                    exec(numpy_setup_str)
                # np_out = np_impl(*np_args)
                np_out = eval("np_impl({})".format(numpy_arg_str), globals())
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

    # # First execution
    # npZ, npN = mandelbrot1_numpy.mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter)
    # nbZ, nbN = mandelbrot1_numba_1.mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter)
    # dc_sdfg = mandelbrot1_dace_1.mandelbrot.to_sdfg()
    # dc_sdfg.apply_transformations_repeated([MapFusion])
    # # dc_sdfg.apply_transformations_repeated([Vectorization])
    # dace_exec = dc_sdfg.compile()
    # # dace_exec = mandelbrot1_dace_1.mandelbrot.compile()
    # dcZ, dcN = dace_exec(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, maxiter=maxiter, horizon=2.0, XN=xn, YN=yn)
    # # daZ, daN = da_impl.mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter)

    # mandelbrot1_numba_1.linspace.parallel_diagnostics(level=4)
    # mandelbrot1_numba_1.mandelbrot.parallel_diagnostics(level=4)

    # # Benchmark
    # time = timeit.repeat(
    #     "mandelbrot1_numpy.mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter)",
    #     setup="pass", repeat=20,number=1, globals=globals())
    # print("NumPy median time: {}".format(np.median(time)))
    # time = timeit.repeat(
    #     "mandelbrot1_numba_1.mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter)",
    #     setup="pass", repeat=20, number=1, globals=globals())
    # print("Numba median time: {}".format(np.median(time)))
    # time = timeit.repeat(
    #     "dace_exec(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, maxiter=maxiter, horizon=2.0, XN=xn, YN=yn)",
    #     setup="pass", repeat=20, number=1, globals=globals())
    # print("DaCe median time: {}".format(np.median(time)))
    # # time = timeit.repeat(
    # #     "da_impl.mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter)",
    # #     setup="pass", repeat=20,number=1, globals=globals())
    # # print("Dask median time: {}".format(np.median(time)))

    # # Validation
    # if not np.allclose(npZ, nbZ):
    #     print("Numba Z relerror: {}".format(np.linalg.norm(npZ - nbZ)/np.linalg.norm(npZ)))
    # if not np.allclose(npN, nbN):
    #     print("Numba N relerror: {}".format(np.linalg.norm(npN - nbN)/np.linalg.norm(npN)))
    # if not np.allclose(npZ, nbZ):
    #     print("DaCe Z relerror: {}".format(np.linalg.norm(npZ - dcZ)/np.linalg.norm(npZ)))
    # if not np.allclose(npN, nbN):
    #     print("DaCe N relerror: {}".format(np.linalg.norm(npN - dcN)/np.linalg.norm(npN)))
