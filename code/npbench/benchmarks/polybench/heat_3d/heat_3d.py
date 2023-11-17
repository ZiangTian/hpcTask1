import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "heat_3d"
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
        arg_str = "TSTEPS, np_A, np_B",
        setup_str = "np_A, np_B = np.copy(A), np.copy(B)",
        report_str = "NumPy",
        out_args = ("np_A", "np_B")
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "TSTEPS, nb_A, nb_B",
        setup_str = "nb_A, nb_B = np.copy(A), np.copy(B)",
        report_str = "Numba",
        out_args = ("nb_A", "nb_B")
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS, pt_A, pt_B",
        setup_str = "pt_A, pt_B = np.copy(A), np.copy(B)",
        report_str = "Pythran",
        out_args = ("pt_A", "pt_B")
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "TSTEPS, gA, gB",
        setup_str = "gA, gB = cp.asarray(A), cp.asarray(B)",
        report_str = "CuPy",
        out_args = ("gA", "gB")
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS=TSTEPS, A=dc_A, B=dc_B, N=N",
        setup_str = "dc_A, dc_B = np.copy(A), np.copy(B)",
        report_str = "DaCe CPU",
        out_args = ("dc_A", "dc_B")
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "TSTEPS=TSTEPS, A=gA, B=gB, N=N",
        setup_str = "gA, gB = cp.asarray(A), cp.asarray(B)",
        report_str = "DaCe GPU",
        out_args = ("gA", "gB")
    )
)


def kernel_orig(TSTEPS, N, A, B):
    
    for t in range(1, TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    B[i, j, k] = (
                        0.125 * (A[i + 1, j, k] - 2.0 * A[i, j, k] +
                                 A[i - 1, j, k]) +
                        0.125 * (A[i, j + 1, k] - 2.0 * A[i, j, k] +
                                 A[i, j - 1, k]) +
                        0.125 * (A[i, j, k + 1] - 2.0 * A[i, j, k] +
                                 A[i, j, k - 1]) +
                        A[i, j, k])
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    A[i, j, k] = (
                        0.125 * (B[i + 1, j, k] - 2.0 * B[i, j, k] +
                                 B[i - 1, j, k]) +
                        0.125 * (B[i, j + 1, k] - 2.0 * B[i, j, k] +
                                 B[i, j - 1, k]) +
                        0.125 * (B[i, j, k + 1] - 2.0 * B[i, j, k] +
                                 B[i, j, k - 1]) +
                        B[i, j, k])


def kernel_numpy(TSTEPS, N, A, B):
    
    for t in range(1, TSTEPS):
        B[1:N-1, 1:N-1, 1:N-1] = (
            0.125 * (A[2:, 1:N-1, 1:N-1] - 2.0 * A[1:N-1, 1:N-1, 1:N-1] +
                     A[:N-2, 1:N-1, 1:N-1]) +
            0.125 * (A[1:N-1, 2:, 1:N-1] - 2.0 * A[1:N-1, 1:N-1, 1:N-1] +
                     A[1:N-1, :N-2, 1:N-1]) +
            0.125 * (A[1:N-1, 1:N-1, 2:] - 2.0 * A[1:N-1, 1:N-1, 1:N-1] +
                     A[1:N-1, 1:N-1, 0:N-2]) +
            A[1:N-1, 1:N-1, 1:N-1])
        A[1:N-1, 1:N-1, 1:N-1] = (
            0.125 * (B[2:, 1:N-1, 1:N-1] - 2.0 * B[1:N-1, 1:N-1, 1:N-1] +
                     B[:N-2, 1:N-1, 1:N-1]) +
            0.125 * (B[1:N-1, 2:, 1:N-1] - 2.0 * B[1:N-1, 1:N-1, 1:N-1] +
                     B[1:N-1, :N-2, 1:N-1]) +
            0.125 * (B[1:N-1, 1:N-1, 2:] - 2.0 * B[1:N-1, 1:N-1, 1:N-1] +
                     B[1:N-1, 1:N-1, 0:N-2]) +
            B[1:N-1, 1:N-1, 1:N-1])


def init_data(N, datatype):

    A = np.empty((N, N, N), dtype=datatype)
    B = np.empty((N, N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A[i, j, k] = B[i, j, k] = (i + j + (N - k)) * 10 / N

    return A, B


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    # TSTEPS, N = 100, 40  # medium dataset
    TSTEPS, N = 500, 120  # large dataset
    # TSTEPS, N = 1000, 200  # extra-large dataset
    A, B = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
