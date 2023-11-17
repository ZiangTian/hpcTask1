import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "gramschmidt"
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
        arg_str = "A=dc_A, M=M, N=N",
        setup_str = "dc_A = np.copy(A)",
        report_str = "DaCe CPU",
        out_args = ("dc_A",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "A=gA, M=M, N=N",
        setup_str = "gA = cp.asarray(A)",
        report_str = "DaCe GPU",
        out_args = ("gA",)
    )
)


def kernel_orig(M, N, A, R, Q):

    for k in range(N):
        nrm = 0.0
        for i in range(M):
            nrm += A[i, k] *A[i, k]
        R[k, k] = np.sqrt(nrm)
        for i in range(M):
            Q[i, k] = A[i, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = 0.0
            for i in range(M):
                R[k, j] += Q[i, k] * A[i, j]
            for i in range(M):
                A[i, j] = A[i, j] - Q[i, k] * R[k, j]


# NOTE: Numpy does QR decompostion with Householder reflections, like LAPACK
def kernel_numpy(M, N, A, R, Q):

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]


def init_data(M, N, datatype):

    A = np.empty((M, N), dtype=datatype)
    R = np.empty((N, N), dtype=datatype)
    Q = np.empty((M, N), dtype=datatype)
    for i in range(M):
        for j in range(N):
            A[i, j] = (((i * j) % M) / M) * 100 + 10
            Q[i, j] = 0.0
    for i in range(N):
        for j in range(N):
            R[i, j] = 0.0

    while np.linalg.matrix_rank(A) < N:
        A = np.random.randn(M, N)

    return A, R, Q


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    M, N = 240, 200  # medium dataset
    # M, N = 1200, 1000  # large dataset
    # M, N = 2600, 2000  # extra-large dataset
    A, R, Q = init_data(M, N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
