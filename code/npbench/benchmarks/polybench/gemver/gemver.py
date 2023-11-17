import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "gemver"
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
        arg_str = "alpha, beta, np_A, u1, v1, u2, v2, np_w, np_x, y, z",
        setup_str = "np_A, np_x, np_w = np.copy(A), np.copy(x), np.copy(w)",
        report_str = "NumPy",
        out_args = ("np_A", "np_w", "np_x")
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, beta, nb_A, u1, v1, u2, v2, nb_w, nb_x, y, z",
        setup_str = "nb_A, nb_x, nb_w = np.copy(A), np.copy(x), np.copy(w)",
        report_str = "Numba",
        out_args = ("nb_A", "nb_w", "nb_x")
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, beta, pt_A, u1, v1, u2, v2, pt_w, pt_x, y, z",
        setup_str = "pt_A, pt_x, pt_w = np.copy(A), np.copy(x), np.copy(w)",
        report_str = "Pythran",
        out_args = ("pt_A", "pt_w", "pt_x")
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, beta, gA, gu1, gu2, gv1, gv2, gw, gx, gy, gz",
        setup_str = "gA, gu1, gu2, gv1, gv2, gw, gx, gy, gz = cp.asarray(A), "
                    "cp.asarray(u1), cp.asarray(u2), cp.asarray(v1), "
                    "cp.asarray(v2), cp.asarray(w), cp.asarray(x), "
                    "cp.asarray(y), cp.asarray(z)",
        report_str = "CuPy",
        out_args = ("gA", "gw", "gx")
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, beta=beta, A=dc_A, u1=u1, v1=v1, u2=u2, "
                  "v2=v2, w=dc_w, x=dc_x, y=y, z=z, N=N",
        setup_str = "dc_A, dc_x, dc_w = np.copy(A), np.copy(x), np.copy(w)",
        report_str = "DaCe CPU",
        out_args = ("dc_A", "dc_w", "dc_x")
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, beta=beta, A=gA, u1=gu1, u2=gu2, v1=gv1, "
                  "v2=gv2, w=gw, x=gx, y=gy, z=gz, N=N",
        setup_str = "gA, gu1, gu2, gv1, gv2, gw, gx, gy, gz = cp.asarray(A), "
                    "cp.asarray(u1), cp.asarray(u2), cp.asarray(v1), "
                    "cp.asarray(v2), cp.asarray(w), cp.asarray(x), "
                    "cp.asarray(y), cp.asarray(z)",
        report_str = "DaCe GPU",
        out_args = ("gA", "gw", "gx")
    )
)


def kernel_orig(N, alpha, beta, A, u1, v1, u2, v2, w, x, y, z):

    for i in range(N):
        for j in range(N):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for i in range(N):
        for j in range(N):
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in range(N):
        x[i] = x[i] + z[i]
    
    for i in range(N):
        for j in range(N):
            w[i] = w[i] + alpha * A[i, j] * x[j]


def kernel_numpy(N, alpha, beta, A, u1, v1, u2, v2, w, x, y, z):

    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A
    x += z
    w += alpha * A @ x


def init_data(N, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.empty((N, N), dtype=datatype)
    u1 = np.empty((N, ), dtype=datatype)
    u2 = np.empty((N, ), dtype=datatype)
    v1 = np.empty((N, ), dtype=datatype)
    v2 = np.empty((N, ), dtype=datatype)
    w = np.empty((N, ), dtype=datatype)
    x = np.empty((N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    z = np.empty((N, ), dtype=datatype)
    for i in range(N):
        u1[i] = i
        u2[i] = ((i + 1) / fn) / 2.0
        v1[i] = ((i + 1) / fn) / 4.0
        v2[i] = ((i + 1) / fn) / 6.0
        y[i] = ((i + 1) / fn) / 8.0
        z[i] = ((i + 1) / fn) / 9.0
        x[i] = 0.0
        w[i] = 0.0
        for j in range(N):
            A[i, j] = (i * j % N) / N

    return alpha, beta, A, u1, u2, v1, v2, w, x, y, z


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
    N= 8000 #XXL dataset?
    alpha, beta, A, u1, u2, v1, v2, w, x, y, z = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
