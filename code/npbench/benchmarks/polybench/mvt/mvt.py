import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "mvt"
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
        arg_str = "np_x1, np_x2, y_1, y_2, A",
        setup_str = "np_x1, np_x2 = np.copy(x1), np.copy(x2)",
        report_str = "NumPy",
        out_args = ("np_x1", "np_x2")
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "nb_x1, nb_x2, y_1, y_2, A",
        setup_str = "nb_x1, nb_x2 = np.copy(x1), np.copy(x2)",
        report_str = "Numba",
        out_args = ("nb_x1", "nb_x2")
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "pt_x1, pt_x2, y_1, y_2, A",
        setup_str = "pt_x1, pt_x2 = np.copy(x1), np.copy(x2)",
        report_str = "Pythran",
        out_args = ("pt_x1", "pt_x2")
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gx1, gx2, gy_1, gy_2, gA",
        setup_str = "gx1, gx2, gy_1, gy_2, gA = cp.asarray(x1), cp.asarray(x2), "
                    "cp.asarray(y_1), cp.asarray(y_2), cp.asarray(A)",
        report_str = "CuPy",
        out_args = ("gx1", "gx2")
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "x1=dc_x1, x2=dc_x2, y_1=y_1, y_2=y_2, A=A, N=N",
        setup_str = "dc_x1, dc_x2 = np.copy(x1), np.copy(x2)",
        report_str = "DaCe CPU",
        out_args = ("dc_x1", "dc_x2")
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "x1=gx1, x2=gx2, y_1=gy_1, y_2=gy_2, A=gA, N=N",
        setup_str = "gx1, gx2, gy_1, gy_2, gA = cp.asarray(x1), cp.asarray(x2), "
                    "cp.asarray(y_1), cp.asarray(y_2), cp.asarray(A)",
        report_str = "DaCe GPU",
        out_args = ("gx1", "gx2")
    )
)


def kernel_orig(N, x1, x2, y_1, y_2, A):

    for i in range(N):
        for j in range(N):
            x1[i] = x1[i] + A[i, j] * y_1[j]
    for i in range(N):
        for j in range(N):
            x2[i] = x2[i] + A[j, i] * y_2[j]


def kernel_numpy(N, x1, x2, y_1, y_2, A):

    x1 += A @ y_1
    x2 += y_2 @ A

def init_data(N, datatype):

    x1 = np.empty((N, ), dtype=datatype)
    x2 = np.empty((N, ), dtype=datatype)
    y_1 = np.empty((N, ), dtype=datatype)
    y_2 = np.empty((N, ), dtype=datatype)
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        x1[i] = (i % N) / N
        x2[i] = ((i + 1) % N) / N
        y_1[i] = ((i + 3) % N) / N
        y_2[i] = ((i + 4) % N) / N
        for j in range(N):
            A[i, j] = (i * j % N) / N

    return x1, x2, y_1, y_2, A


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
    x1, x2, y_1, y_2, A = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
