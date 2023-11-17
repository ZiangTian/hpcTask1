import argparse
import pathlib
import numpy as np
from npbench import run, run_dace, run_numba, str2bool


# Module name
module_name = "contour_integral"
func_name = "contour_integral"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "qts",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "NR, NM, slab_per_bc, Ham, int_pts, Y",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "NR, NM, slab_per_bc, Ham, int_pts, Y",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "NR, NM, slab_per_bc, Ham, int_pts, Y",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "NR, NM, slab_per_bc, gHam, gint_pts, gY",
        setup_str = "gHam, gint_pts, gY = cp.asarray(Ham), cp.asarray(int_pts), cp.asarray(Y)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "Ham=Ham, int_pts=int_pts, Y=Y, NR=NR, NM=NM, slab_per_bc=slab_per_bc",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "Ham=gHam, int_pts=gint_pts, Y=gY, NR=NR, NM=NM, slab_per_bc=slab_per_bc",
        setup_str = "gHam, gint_pts, gY = cp.asarray(Ham), cp.asarray(int_pts), cp.asarray(Y)",
        report_str = "DaCe GPU"
    )
)

# # Argument strings
# numpy_arg_str = "NR, NM, slab_per_bc, Ham, int_pts, Y"
# numba_arg_str = "NR, NM, slab_per_bc, Ham, int_pts, Y"
# # dace_arg_str = "nx=nx, ny=ny, nit=nit, u=dc_u, v=dc_v, dt=dt, dx=dx, dy=dy, p=dc_p, rho=rho, nu=nu, F=F"
# # cupy_arg_str = "nit, cp_u, cp_v, dt, dx, dy, cp_p, rho, nu, F"

# # Setup string
# numpy_setup_str = "pass"
# numba_setup_str = "pass"
# dace_setup_str = "pass"

# # Report string
# numpy_report_str = "NumPy"
# numba_report_str = "Numba"
# dace_report_str = "DaCe"
# cupy_report_str = "CuPy"


def rng_complex(shape):
    return (np.random.rand(*shape).astype(np.float64) +
            np.random.rand(*shape).astype(np.float64) * 1j)


def initialize(NR, NM, slab_per_bc, num_int_pts):
    Ham = rng_complex((slab_per_bc + 1, NR, NR))
    int_pts = rng_complex((num_int_pts, ))
    Y = rng_complex((NR, NM))
    return Ham, int_pts, Y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    NR = 500
    NM = 1000
    slab_per_bc = 2
    num_int_pts = 32
    Ham, int_pts, Y = initialize(NR, NM, slab_per_bc, num_int_pts)

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
    #             from contour_integral_numpy import contour_integral as np_impl
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
