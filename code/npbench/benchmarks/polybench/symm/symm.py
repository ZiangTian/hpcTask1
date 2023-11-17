import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "symm"
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
        arg_str = "alpha, beta, np_C, A, B",
        setup_str = "np_C = np.copy(C)",
        report_str = "NumPy",
        out_args = ("np_C",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, beta, nb_C, A, B",
        setup_str = "nb_C = np.copy(C)",
        report_str = "Numba",
        out_args = ("nb_C",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, beta, pt_C, A, B",
        setup_str = "pt_C = np.copy(C)",
        report_str = "Pythran",
        out_args = ("pt_C",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, beta, gC, gA, gB",
        setup_str = "gC, gA, gB = cp.asarray(C), cp.asarray(A), cp.asarray(B)",
        report_str = "CuPy",
        out_args = ("gC",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, beta=beta, A=A, B=B, C=dc_C, M=M, N=N",
        setup_str = "dc_C = np.copy(C)",
        report_str = "DaCe CPU",
        out_args = ("dc_C",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, beta=beta, A=gA, B=gB, C=gC, M=M, N=N",
        setup_str = "gC, gA, gB = cp.asarray(C), cp.asarray(A), cp.asarray(B)",
        report_str = "DaCe GPU",
        out_args = ("gC",)
    )
)


def kernel_orig(M, N, alpha, beta, C, A, B):

    for i in range(M):
        for j in range(N):
            temp2 = 0
            for k in range(i):
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2


def kernel_numpy(M, N, alpha, beta, C, A, B):

    temp2 = np.empty((N, ), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


def init_data(M, N, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.empty((M, N), dtype=datatype)
    A = np.empty((M, M), dtype=datatype)
    B = np.empty((M, N), dtype=datatype)
    for i in range(M):
        for j in range(N):
            C[i, j] = ((i + j) % 100) / M
            B[i, j] = ((N + i - j) % 100) / M
    for i in range(M):
        for j in range(i + 1):
            A[i, j] = ((i + j) % 100) / M
        for j in range(i + 1, M):
            A[i, j] = -999

    return alpha, beta, C, A, B


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
    # M, N = 2000, 2600 # extra-large dataset
    alpha, beta, C, A, B = init_data(M, N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
