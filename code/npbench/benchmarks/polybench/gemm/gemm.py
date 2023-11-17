import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "gemm"
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
        arg_str = "alpha, beta, np_C, A, B",
        setup_str = "np_C = np.copy(C)",
        report_str = "NumPy",
        out_args = ("np_C",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, beta, nb_C, A, B",
        setup_str = "nb_C = np.copy(C)",
        report_str = "Numba",
        out_args = ("nb_C",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, beta, pt_C, A, B",
        setup_str = "pt_C = np.copy(C)",
        report_str = "Pythran",
        out_args = ("pt_C",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, beta, gC, gA, gB",
        # setup_str = "galpha, gbeta, gA, gB, gC = cp.asarray(alpha), "
        #             "cp.asarray(beta), cp.asarray(A), cp.asarray(B), cp.asarray(C)",
        setup_str = "gA, gB, gC = cp.asarray(A), cp.asarray(B), cp.asarray(C)",
        report_str = "CuPy",
        out_args = ("gC",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, beta=beta, A=A, B=B, C=dc_C, NI=NI, NJ=NJ, NK=NK",
        setup_str = "dc_C = np.copy(C)",
        report_str = "DaCe CPU",
        out_args = ("dc_C",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, beta=beta, A=gA, B=gB, C=gC, NI=NI, NJ=NJ, NK=NK",
        # setup_str = "gA, gB, gC = cuda.to_device(A), cuda.to_device(B), cuda.to_device(C)",
        setup_str = "gA, gB, gC = cp.asarray(A), cp.asarray(B), cp.asarray(C)",
        report_str = "DaCe GPU",
        out_args = ("gC",)
    ),
    legate = dict(
        module_str = "{}_legate".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, beta, lg_C, A, B",
        setup_str = "lg_C = np.copy(C)",
        report_str = "Legate CPU",
        out_args = ("lg_C",)
    ),
)


def kernel_orig(NI, NJ, NK, alpha, beta, C, A, B):

    for i in range(NI):
        for j in range(NJ):
            C[i, j] *= beta
            # C[i, j] = C[i, j] * beta
        for k in range(NK):
            for j in range(NJ):
                C[i, j] += alpha * A[i, k] * B[k, j]
                # C[i, j] = C[i, j] + alpha * A[i, k] * B[k, j]


def kernel_numpy(NI, NJ, NK, alpha, beta, C, A, B):

    # C = alpha * A @ B + beta * C 
    C *= beta
    C += alpha * A @ B


def init_data(NI, NJ, NK, datatype):

    from dace.libraries.standard.memory import aligned_ndarray

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = aligned_ndarray(np.empty((NI, NJ), dtype=datatype), alignment=4096)
    for i in range(NI):
        for j in range(NJ):
            C[i, j] = ((i * j + 1) % NI) / NI
    A = aligned_ndarray(np.empty((NI, NK), dtype=datatype), alignment=4096)
    for i in range(NI):
        for k in range(NK):
            A[i, k] = (i * (k + 1) % NK) / NK
    B = aligned_ndarray(np.empty((NK, NJ), dtype=datatype), alignment=4096)
    for k in range(NK):
        for j in range(NJ):
            C[i, j] = (k * (j + 2) % NJ) / NJ

    return alpha, beta, C, A, B


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    NI, NJ, NK = 2000, 2300, 2600  # extra-large dataset
    # NI, NJ, NK = 4000, 4600, 5200  XXL dataset?
    alpha, beta, C, A, B = init_data(NI, NJ, NK, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
