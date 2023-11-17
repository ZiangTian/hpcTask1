import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "trmm"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "kernels",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, A, np_B",
        setup_str = "np_B = np.copy(B)",
        report_str = "NumPy",
        out_args = ("np_B",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, A, nb_B",
        setup_str = "nb_B = np.copy(B)",
        report_str = "Numba",
        out_args = ("nb_B",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, A, pt_B",
        setup_str = "pt_B = np.copy(B)",
        report_str = "Pythran",
        out_args = ("pt_B",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, gA, gB",
        setup_str = "gA, gB = cp.asarray(A), cp.asarray(B)",
        report_str = "CuPy",
        out_args = ("gB",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, A=A, B=dc_B, M=M, N=N",
        setup_str = "dc_B = np.copy(B)",
        report_str = "DaCe CPU",
        out_args = ("dc_B",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, A=gA, B=gB, M=M, N=N",
        setup_str = "gA, gB = cp.asarray(A), cp.asarray(B)",
        report_str = "DaCe GPU",
        out_args = ("gB",)
    )
)


def kernel_orig(M, N, alpha, A, B):

    for i in range(M):
        for j in range(N):
            for k in range(i + 1, M):
                B[i, j] += A[k, i] * B[k, j]
            B[i, j] = alpha * B[i, j]


def kernel_numpy(M, N, alpha, A, B):

    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha


def init_data(M, N, datatype):

    alpha = datatype(1.5)
    A = np.empty((M, M), dtype=datatype)
    B = np.empty((M, N), dtype=datatype)
    for i in range(M):
        for j in range(M):
            A[i, j] = ((i * j) % M) / M
        A[i, i] = 1.0
        for j in range(N):
            B[i, j] = ((N + i - j) % N) / N

    return alpha, A, B


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    M, N = 1000, 1200  # large dataset
    # M, N = 2000, 2600  # extra-large dataset
    alpha, A, B = init_data(M, N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())

