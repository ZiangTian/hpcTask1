import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "bicg"
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
        arg_str = "A, p, r",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "A, p, r",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A, p, r",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gA, gp, gr",
        setup_str = "gA, gp, gr = cp.asarray(A), cp.asarray(p), cp.asarray(r)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A=A, p=p, r=r, M=M, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "A=gA, p=gp, r=gr, M=M, N=N",
        setup_str = "gA, gp, gr = cp.asarray(A), cp.asarray(p), cp.asarray(r)",
        report_str = "DaCe GPU"
    )
)


def kernel_orig(M, N, A, s, q, p, r):

    for i in range(N):
        s[i] = 0.0
    for i in range(N):
        q[i] = 0.0
        for j in range(M):
            s[j] = s[j] + r[i] * A[i, j]
            q[i] = q[i] + A[i, j] * p[j]


def kernel_numpy(M, N, A, s, q, p, r):

    s[:] = r @ A
    q[:] = A @ p


def init_data(M, N, datatype):

    A = np.empty((N, M), dtype=datatype)
    s = np.empty((M, ), dtype=datatype)
    q = np.empty((N, ), dtype=datatype)
    p = np.empty((M, ), dtype=datatype)
    r = np.empty((N, ), dtype=datatype)
    for i in range(M):
        p[i] = (i % M) / M
    for i in range(N):
        r[i] = (i % N) / N
        for j in range(M):
            A[i, j] = (i * (j + 1) % N) / N

    return A, s, q, p, r


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    # M, N = 1800, 2200
    M, N = 18000, 22000  # XXL dataset? Which devices does it fit?
    # A will be ~3GB (float64)
    A, s, q, p, r = init_data(M, N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
