import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "durbin"
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
        arg_str = "r",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "r",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "r",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gr",
        setup_str = "gr = cp.asarray(r)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "r=r, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "r=gr, N=N",
        setup_str = "gr = cp.asarray(r)",
        report_str = "DaCe GPU"
    )
)


def kernel_orig(N, r, y):

    z = np.empty((N, ), dtype=r.dtype)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, N):
        beta = (1.0 - alpha * alpha) * beta
        sum = 0.0
        for i in range(k):
            sum += r[k - i - 1] * y[i]
        alpha = - (r[k] + sum) / beta
        for i in range(k):
            z[i] = y[i] + alpha * y[k - i - 1]
        for i in range(k):
            y[i] = z[i]
        y[k] = alpha


def kernel_numpy(N, r, y):

    z = np.empty((N, ), dtype=r.dtype)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, N):
        beta = (1.0 - alpha * alpha) * beta
        sum = np.dot(np.flip(r[:k]), y[:k])
        alpha = - (r[k] + sum) / beta
        z[:k] = y[:k] + alpha * np.flip(y[:k])
        y[:k] = z[:k]
        y[k] = alpha


def kernel_numpy2(r):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = - (r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha

    return y


def init_data(N, datatype):

    r = np.empty((N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    for i in range(N):
        r[i] = N + 1 - i

    return r, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    #N = 4000  # extra-large dataset
    N= 16000 
    r, y = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
