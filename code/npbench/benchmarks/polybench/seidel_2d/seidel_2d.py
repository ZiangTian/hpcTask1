import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "seidel_2d"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "kernels",
    dwarf = "structured_grids",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS, N, np_A",
        setup_str = "np_A = np.copy(A)",
        report_str = "NumPy",
        out_args = ("np_A",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "TSTEPS, N, nb_A",
        setup_str = "nb_A = np.copy(A)",
        report_str = "Numba",
        out_args = ("nb_A",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS, N, pt_A",
        setup_str = "pt_A = np.copy(A)",
        report_str = "Pythran",
        out_args = ("pt_A",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "TSTEPS, N, gA",
        setup_str = "gA = cp.asarray(A)",
        report_str = "CuPy",
        out_args = ("gA",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS=TSTEPS, A=dc_A, N=N",
        setup_str = "dc_A = np.copy(A)",
        report_str = "DaCe CPU",
        out_args = ("dc_A",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "TSTEPS=TSTEPS, A=gA, N=N",
        setup_str = "gA = cp.asarray(A)",
        report_str = "DaCe GPU",
        out_args = ("gA",)
    )
)


def kernel_orig(TSTEPS, N, A):
    
    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                A[i, j] = (A[i - 1, j - 1] + A[i - 1, j] + A[i - 1, j + 1] +
                           A[i, j - 1] + A[i, j] + A[i, j + 1] +
                           A[i + 1, j - 1] + A[i + 1, j] + A[i + 1, j + 1]
                          ) / 9.0


def kernel_numpy(TSTEPS, N, A):
    
    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:N-1] += (A[i-1, :N-2] + A[i-1, 1:N-1] + A[i-1, 2:] +
                            A[i, 2:] + A[i+1, :N-2] + A[i+1, 1:N-1] +
                            A[i+1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


def init_data(N, datatype):

    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            A[i, j] = (i * (j + 2) + 2) / N

    return A


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    TSTEPS, N = 100, 400  # medium dataset
    # TSTEPS, N = 500, 2000  # large dataset
    # TSTEPS, N = 1000, 4000  # extra-large dataset
    A = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
