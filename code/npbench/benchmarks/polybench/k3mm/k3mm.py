import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "k3mm"
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
        arg_str = "A, B, C, D",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "A, B, C, D",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A, B, C, D",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gA, gB, gC, gD",
        setup_str = "gA, gB, gC, gD = cp.asarray(A), cp.asarray(B), cp.asarray(C), cp.asarray(D)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A=A, B=B, C=C, D=D, NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "A=gA, B=gB, C=gC, D=gD, NI=NI, NJ=NJ, NK=NK, NL=NL, NM=NM",
        setup_str = "gA, gB, gC, gD = cp.asarray(A), cp.asarray(B), cp.asarray(C), cp.asarray(D)",
        report_str = "DaCe GPU"
    )
)


def kernel_orig(NI, NJ, NK, NL, NM, E, A, B, F, C, D, G):

    for i in range(NI):
        for j in range(NJ):
            E[i, j] = 0.0
            for k in range(NK):
                E[i, j] += A[i, k] * B[k, j]
    for i in range(NJ):
        for j in range(NL):
            F[i, j] = 0.0
            for k in range(NM):
                F[i, j] += C[i, k] * D[k, j]
    for i in range(NI):
        for j in range(NL):
            G[i, j] = 0.0
            for k in range(NJ):
                G[i, j] += E[i, k] * F[k, j]


def kernel_numpy(NI, NJ, NK, NL, NM, E, A, B, F, C, D, G):

    E[:] = A @ B
    F[:] = C @ D
    G[:] = E @ F


def init_data(NI, NJ, NK, NL, NM, datatype):

    E = np.empty((NI, NJ), dtype=datatype)
    A = np.empty((NI, NK), dtype=datatype)
    B = np.empty((NK, NJ), dtype=datatype)
    F = np.empty((NJ, NL), dtype=datatype)
    C = np.empty((NJ, NM), dtype=datatype)
    D = np.empty((NM, NL), dtype=datatype)
    G = np.empty((NI, NL), dtype=datatype)
    for i in range(NI):
        for j in range(NK):
            A[i, j] = ((i * j + 1) % NI) / (5 * NI)
    for i in range(NK):
        for j in range(NJ):
            B[i, j] = ((i * (j + 1) + 2) % NJ) / (5 * NJ)
    for i in range(NJ):
        for j in range(NM):
            C[i, j] = (i * (j + 3) % NL) / (5 * NL)
    for i in range(NM):
        for j in range(NL):
            D[i, j] = ((i * (j + 2) + 2) % NK) / ( 5 * NK)

    return E, A, B, F, C, D, G


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    # NI, NJ, NK, NL, NM = 1600, 1800, 2000, 2200, 2400  # extra-large dataset
    NI, NJ, NK, NL, NM = 3200, 3600, 4000, 4400, 4800  # XXL dataset?
    E, A, B, F, C, D, G = init_data(NI, NJ, NK, NL, NM , np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
