import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "lu"
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
        arg_str = "np_A",
        setup_str = "np_A = np.copy(A)",
        report_str = "NumPy",
        out_args = ("np_A",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "nb_A",
        setup_str = "nb_A = np.copy(A)",
        report_str = "Numba",
        out_args = ("nb_A",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "pt_A",
        setup_str = "pt_A = np.copy(A)",
        report_str = "Pythran",
        out_args = ("pt_A",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gA",
        setup_str = "gA = cp.asarray(A)",
        report_str = "CuPy",
        out_args = ("gA",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A=dc_A, N=N",
        setup_str = "dc_A = np.copy(A)",
        report_str = "DaCe CPU",
        out_args = ("dc_A",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "A=gA, N=N",
        setup_str = "gA = cp.asarray(A)",
        report_str = "DaCe GPU",
        out_args = ("gA",)
    )
)


def kernel_orig(N, A):

    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i, j] -= A[i, k] * A[k, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            for k in range(i):
                A[i, j] -= A[i, k] * A[k, j]


def kernel_numpy(N, A):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]


def init_data(N, datatype):

    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        for j in range(i + 1):
            A[i, j] = (-j % N) / N + 1
        for j in range(i + 1, N):
            A[i, j] = 0.0
        A[i, i] = 1.0

    B = np.empty((N, N), dtype=datatype)
    # for r in range(N):
    #     for s in range(N):
    #         B[r, s] = 0.0
    # for t in range(N):
    #     for r in range(N):
    #         for s in range(N):
    #             B[r, s] += A[r, t] * A[s, t]
    # for r in range(N):
    #     for s in range(N):
    #         A[r, s] = B[r, s]
    B[:] = A @ np.transpose(A)
    A[:] = B

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
    N = 2000  # large dataset
    # N = 4000  # extra-large dataset
    A = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
