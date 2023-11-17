import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "k2mm"
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
        arg_str = "alpha, beta, A, B, C, np_D",
        setup_str = "np_D = np.copy(D)",
        report_str = "NumPy",
        out_args = ("np_D",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, beta, A, B, C, nb_D",
        setup_str = "nb_D = np.copy(D)",
        report_str = "Numba",
        out_args = ("nb_D",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, beta, A, B, C, pt_D",
        setup_str = "pt_D = np.copy(D)",
        report_str = "Pythran",
        out_args = ("pt_D",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, beta, gA, gB, gC, gD",
        setup_str = "gA, gB, gC, gD = cp.asarray(A), cp.asarray(B), cp.asarray(C), cp.asarray(D)",
        report_str = "CuPy",
        out_args = ("gD",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, beta=beta, A=A, B=B, C=C, D=dc_D, "
                  "NI=NI, NJ=NJ, NK=NK, NL=NL",
        setup_str = "dc_D = np.copy(D)",
        report_str = "DaCe CPU",
        out_args = ("dc_D",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, beta=beta, A=gA, B=gB, C=gC, D=gD, "
                  "NI=NI, NJ=NJ, NK=NK, NL=NL",
        setup_str = "gA, gB, gC, gD = cp.asarray(A), cp.asarray(B), cp.asarray(C), cp.asarray(D)",
        report_str = "DaCe GPU",
        out_args = ("gD",)
    )
)


def kernel_orig(NI, NJ, NK, NL, alpha, beta, tmp, A, B, C, D):

    for i in range(NI):
        for j in range(NJ):
            tmp[i, j] = 0.0
            for k in range(NK):
                tmp[i, j] += alpha * A[i, k] * B[k, j]
    for i in range(NI):
        for j in range(NL):
            D[i, j] *= beta
            for k in range(NJ):
                D[i, j] += tmp[i, k] * C[k, j]


def kernel_numpy(NI, NJ, NK, NL, alpha, beta, tmp, A, B, C, D):

    tmp[:] = alpha * A @ B
    D *= beta
    D += tmp @ C
    # D += alpha * A @ B @ C


def init_data(NI, NJ, NK, NL, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    tmp = np.empty((NI, NJ), dtype=datatype)
    A = np.empty((NI, NK), dtype=datatype)
    B = np.empty((NK, NJ), dtype=datatype)
    C = np.empty((NJ, NL), dtype=datatype)
    D = np.empty((NI, NL), dtype=datatype)
    for i in range(NI):
        for j in range(NK):
            A[i, j] = ((i * j + 1) % NI) / NI
    for i in range(NK):
        for j in range(NJ):
            B[i, j] = (i * (j + 1) % NJ) / NJ
    for i in range(NJ):
        for j in range(NL):
            C[i, j] = ((i * (j + 3) + 1) % NL) / NL
    for i in range(NI):
        for j in range(NL):
            D[i, j] = (i * (j + 2) % NK) / NK

    return alpha, beta, tmp, A, B, C, D


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    # NI, NJ, NK, NL = 1600, 1800, 2200, 2400  # extra-large dataset
    NI, NJ, NK, NL = 3200, 3600, 4400, 4800  # XXL dataset?
    alpha, beta, tmp, A, B, C, D = init_data(NI, NJ, NK, NL, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
