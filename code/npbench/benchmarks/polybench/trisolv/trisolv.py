import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "trisolv"
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
        arg_str = "L, x, np_b",
        setup_str = "np_b = np.copy(b)",
        report_str = "NumPy",
        out_args = ("np_b",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "L, x, nb_b",
        setup_str = "nb_b = np.copy(b)",
        report_str = "Numba",
        out_args = ("nb_b",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "L, x, pt_b",
        setup_str = "pt_b = np.copy(b)",
        report_str = "Pythran",
        out_args = ("pt_b",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gL, gx, gb",
        setup_str = "gL, gx, gb = cp.asarray(L), cp.asarray(x), cp.asarray(b)",
        report_str = "CuPy",
        out_args = ("gb",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "L=L, x=x, b=dc_b, N=N",
        setup_str = "dc_b = np.copy(b)",
        report_str = "DaCe CPU",
        out_args = ("dc_b",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "L=gL, x=gx, b=gb, N=N",
        setup_str = "gL, gx, gb = cp.asarray(L), cp.asarray(x), cp.asarray(b)",
        report_str = "DaCe GPU",
        out_args = ("gb",)
    )
)


def kernel_orig(N, L, x, b):

    for i in range(N):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] = x[i] / L[i, i]


def kernel_numpy(N, L, x, b):

    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


def init_data(N, datatype):

    L = np.empty((N, N), dtype=datatype)
    x = np.empty((N, ), dtype=datatype)
    b = np.empty((N, ), dtype=datatype)
    for i in range(N):
        x[i] = -999
        b[i] = i
        for j in range(i + 1):
            L[i, j] = (i + N - j + 1) * 2 / N

    return L, x, b


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    # N = 4000  # extra-large dataset
    N = 16000  # XXL dataset?
    L, x, b = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
