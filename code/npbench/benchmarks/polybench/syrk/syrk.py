import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "syrk"
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
        arg_str = "alpha, beta, np_C, A",
        setup_str = "np_C = np.copy(C)",
        report_str = "NumPy",
        out_args = ("np_C",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, beta, nb_C, A",
        setup_str = "nb_C = np.copy(C)",
        report_str = "Numba",
        out_args = ("nb_C",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, beta, pt_C, A",
        setup_str = "pt_C = np.copy(C)",
        report_str = "Pythran",
        out_args = ("pt_C",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, beta, gC, gA",
        setup_str = "gC, gA = cp.asarray(C), cp.asarray(A)",
        report_str = "CuPy",
        out_args = ("gC",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, beta=beta, A=A, C=dc_C, M=M, N=N",
        setup_str = "dc_C = np.copy(C)",
        report_str = "DaCe CPU",
        out_args = ("dc_C",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, beta=beta, A=gA, C=gC, M=M, N=N",
        setup_str = "gC, gA = cp.asarray(C), cp.asarray(A)",
        report_str = "DaCe GPU",
        out_args = ("gC",)
    )
)


def kernel_orig(N, M, alpha, beta, C, A):

    for i in range(N):
        for j in range(i + 1):
            C[i, j] *= beta
        for k in range(M):
            for j in range(i + 1):
                C[i, j] += alpha * A[i, k] * A[j, k]


def kernel_numpy(N, M, alpha, beta, C, A):

    for i in range(N):
        C[i, :i + 1] *= beta 
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]


def init_data(N, M, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.empty((N, N), dtype=datatype)
    A = np.empty((N, M), dtype=datatype)
    for i in range(N):
        for j in range(M):
            A[i, j] = ((i * j + 1) % N) / N
    for i in range(N):
        for j in range(N):
            C[i, j] = ((i * j + 2) % M) / M

    return alpha, beta, C, A


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    N, M = 1200, 1000  # large dataset
    # N, M = 2600, 2000 # extra-large dataset
    alpha, beta, C, A = init_data(N, M, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
