import argparse
import pathlib
import numpy as np
from npbench import run, run_dace, run_numba, str2bool


# Module name
module_name = "cavity_flow"
func_name = "cavity_flow"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "cfds",
    dwarf = "structured_grids",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "nx, ny, nt, nit, np_u, np_v, dt, dx, dy, np_p, rho, nu",
        setup_str = "np_u, np_v, np_p = np.copy(u), np.copy(v), np.copy(p)",
        report_str = "NumPy",
        out_args = ("np_u", "np_v", "np_p")
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "nx, ny, nt, nit, nb_u, nb_v, dt, dx, dy, nb_p, rho, nu",
        setup_str = "nb_u, nb_v, nb_p = np.copy(u), np.copy(v), np.copy(p)",
        report_str = "Numba",
        out_args = ("nb_u", "nb_v", "nb_p")
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "nx, ny, nt, nit, pt_u, pt_v, dt, dx, dy, pt_p, rho, nu",
        setup_str = "pt_u, pt_v, pt_p = np.copy(u), np.copy(v), np.copy(p)",
        report_str = "Pythran",
        out_args = ("pt_u", "pt_v", "pt_p")
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "nx, ny, nt, nit, gu, gv, dt, dx, dy, gp, rho, nu",
        setup_str = "gu, gv, gp = cp.asarray(u), cp.asarray(v), cp.asarray(p)",
        report_str = "CuPy",
        out_args = ("gu", "gv", "gp")
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "nx=nx, ny=ny, nt=nt, nit=nit, u=dc_u, v=dc_v, dt=dt, dx=dx, "
                  "dy=dy, p=dc_p, rho=rho, nu=nu",
        setup_str = "dc_u, dc_v, dc_p = np.copy(u), np.copy(v), np.copy(p)",
        report_str = "DaCe CPU",
        out_args = ("dc_u", "dc_v", "dc_p")
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "nx=nx, ny=ny, nt=nt, nit=nit, u=gu, v=gv, dt=dt, dx=dx, "
                  "dy=dy, p=gp, rho=rho, nu=nu",
        setup_str = "gu, gv, gp = cp.asarray(u), cp.asarray(v), cp.asarray(p)",
        report_str = "DaCe GPU",
        out_args = ("gu", "gv", "gp")
    )
)

# # Argument strings
# numpy_arg_str = "nx, ny, nt, nit, np_u, np_v, dt, dx, dy, np_p, rho, nu"
# numba_arg_str = "nx, ny, nt, nit, nb_u, nb_v, dt, dx, dy, nb_p, rho, nu"
# dace_arg_str = "nx=nx, ny=ny, nt=nt, nit=nit, u=dc_u, v=dc_v, dt=dt, dx=dx, dy=dy, p=dc_p, rho=rho, nu=nu"
# cupy_arg_str = "nx, ny, nt, nit, cp_u, cp_v, dt, dx, dy, cp_p, rho, nu"

# # Setup string
# numpy_setup_str = "np_u, np_v, np_p = np.copy(u), np.copy(v), np.copy(p)"
# numba_setup_str = "nb_u, nb_v, nb_p = np.copy(u), np.copy(v), np.copy(p)"
# dace_setup_str = "dc_u, dc_v, dc_p = np.copy(u), np.copy(v), np.copy(p)"

# # Report string
# numpy_report_str = "NumPy"
# numba_report_str = "Numba"
# dace_report_str = "DaCe"
# cupy_report_str = "CuPy"


def initialize(ny, nx):
    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    p = np.zeros((ny, nx), dtype=np.float64)
    return u, v, p


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    nx = 101  # 41
    ny = 101  # 41
    nt = 700
    nit = 50
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = .001
    rho = 1.
    nu = .1
    u, v, p = initialize(ny, nx)

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
    #             from cavity_flow_numpy import cavity_flow as np_impl
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
