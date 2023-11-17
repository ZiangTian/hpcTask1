import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "atax"
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
        arg_str = "A, x",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "A, x",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A, x",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gA, gx",
        setup_str = "gA, gx = cp.asarray(A), cp.asarray(x)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A=A, x=x, M=M, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "A=gA, x=gx, M=M, N=N",
        # setup_str = "gA, gx = cuda.to_device(A), cuda.to_device(x)",
        setup_str = "gA, gx = cp.asarray(A), cp.asarray(x)",
        report_str = "DaCe GPU"
    )
)


def kernel_orig(M, N, A, x, y, tmp):

    for i in range(N):
        y[i] = 0.0
    for i in range(M):
        tmp[i] = 0.0
        for j in range(N):
            tmp[i] = tmp[i] + A[i, j] * x[j]
        for j in range(N):
            y[j] = y[j] + A[i, j] * tmp[i]


def kernel_numpy(M, N, A, x, y, tmp):

    tmp[:] = A @ x
    y[:] = tmp @ A


def init_data(M, N, datatype):

    fn = datatype(N)
    A = np.empty((M, N), dtype=datatype)
    x = np.empty((N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    tmp = np.empty((M, ), dtype=datatype)
    for i in range(N):
        x[i] = 1 + (i / fn)
    for i in range(M):
        for j in range(N):
            A[i, j] = ((i + j) % N) / (5 * M)

    return A, x, y, tmp


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    # M, N = 1800, 2200  # extra-large dataset
    M, N = 18000, 22000  # XXL dataset? Which devices does it fit?
    # A will be ~3GB (float64)
    A, x, y, tmp = init_data(M, N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
