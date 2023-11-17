import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "gesummv"
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
        arg_str = "alpha, beta, A, B, x",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, beta, A, B, x",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, beta, A, B, x",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, beta, gA, gB, gx",
        setup_str = "gA, gB, gx = cp.asarray(A), cp.asarray(B), cp.asarray(x)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, beta=beta, A=A, B=B, x=x, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, beta=beta, A=gA, B=gB, x=gx, N=N",
        setup_str = "gA, gB, gx = cp.asarray(A), cp.asarray(B), cp.asarray(x)",
        report_str = "DaCe GPU"
    )
)


def kernel_orig(N, alpha, beta, A, B, tmp, x, y):

    for i in range(N):
        tmp[i] = 0.0
        y[i] = 0.0
        for j in range(N):
            tmp[i] = A[i, j] * x[j] + tmp[i]
            y[i] = B[i, j] * x[j] + y[i]
        y[i] = alpha * tmp[i] + beta * y[i]


def kernel_numpy(N, alpha, beta, A, B, tmp, x, y):

    tmp[:] = A @ x
    y[:] = B @ x
    y *= beta
    y += alpha * tmp


def init_data(N, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.empty((N, N), dtype=datatype)
    B = np.empty((N, N), dtype=datatype)
    tmp = np.empty((N, ), dtype=datatype)
    x = np.empty((N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    for i in range(N):
        x[i] = (i % N) % N
        for j in range(N):
            A[i, j] = ((i * j + 1) % N) / N
            B[i, j] = ((i * j + 2) % N) / N

    return alpha, beta, A, B, tmp, x, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    # N = 2800  # extra-large dataset
    N = 11200
    # 11200 XXL dataset?
    alpha, beta, A, B, tmp, x, y = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
