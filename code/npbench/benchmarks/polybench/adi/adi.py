import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "adi"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "kernels",
    dwarf = "structured_grids",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS, N, np_u",
        setup_str = "np_u = np.copy(u)",
        report_str = "NumPy",
        out_args = ("np_u",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "TSTEPS, N, nb_u",
        setup_str = "nb_u = np.copy(u)",
        report_str = "Numba",
        out_args = ("nb_u",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS, N, pt_u",
        setup_str = "pt_u = np.copy(u)",
        report_str = "Pythran",
        out_args = ("pt_u",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "TSTEPS, N, gu",
        setup_str = "gu = cp.asarray(u)",
        report_str = "CuPy",
        out_args = ("gu",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TSTEPS=TSTEPS, u=dc_u, N=N",
        setup_str = "dc_u = np.copy(u)",
        report_str = "DaCe CPU",
        out_args = ("dc_u",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "TSTEPS=TSTEPS, u=gu, N=N",
        setup_str = "gu = cp.asarray(u)",
        report_str = "DaCe GPU",
        out_args = ("gu",)
    )
)


def kernel_orig(TSTEPS, N, u, v, p, q):

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = - mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = - mul2 / 2.0
    e = 1.0 + mul2
    f = d
    
    for t in range(1, TSTEPS + 1):
        for i in range(1, N - 1):
            v[0, i] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = v[0, i]
            for j in range(1, N - 1):
                p[i, j] = - c / (a * p[i, j - 1] + b)
                q[i, j] = (-d * u[j, i - 1] + (1.0 + 2.0 * d) * u[j, i] -
                           f * u[j, i + 1] - a * q[i, j - 1]) / (
                               a * p[i, j - 1] + b)
            v[N - 1, i] = 1.0
            for j in range(N - 2, 0, -1):
                v[j, i] = p[i, j] * v[j + 1, i] + q[i, j]

        for i in range(1, N - 1):
            u[i, 0] = 1.0
            p[i, 0] = 0.0
            q[i, 0] = u[i, 0]
            for j in range(1, N - 1):
                p[i, j] = - f / (d * p[i, j - 1] + e)
                q[i, j] = (-a * v[i - 1, j] + (1.0 + 2.0 * a) * v[i, j] -
                           c * v[i + 1, j] - d * q[i, j - 1]) / (
                               d * p[i, j - 1] + e)
            u[i, N - 1] = 1.0
            for j in range(N - 2, 0, -1):
                u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]


def kernel_numpy(TSTEPS, N, u, v, p, q):

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = - mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = - mul2 / 2.0
    e = 1.0 + mul2
    f = d
    
    for t in range(1, TSTEPS + 1):
        # for i in range(1, N - 1):
        #     v[0, i] = 1.0
        #     p[i, 0] = 0.0
        #     q[i, 0] = v[0, i]
        #     for j in range(1, N - 1):
        #         p[i, j] = - c / (a * p[i, j - 1] + b)
        #         q[i, j] = (-d * u[j, i - 1] + (1.0 + 2.0 * d) * u[j, i] -
        #                    f * u[j, i + 1] - a * q[i, j - 1]) / (
        #                        a * p[i, j - 1] + b)
        #     v[N - 1, i] = 1.0
        #     for j in range(N - 2, 0, -1):
        #         v[j, i] = p[i, j] * v[j + 1, i] + q[i, j]
        v[0, 1:N - 1] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = v[0, 1:N - 1]
        for j in range(1, N - 1):
            p[1:N - 1, j] = - c / (a * p[1:N - 1, j - 1] + b)
            q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) *
                             u[j, 1:N - 1] - f * u[j, 2:N] - a *
                             q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]

        # for i in range(1, N - 1):
        #     u[i, 0] = 1.0
        #     p[i, 0] = 0.0
        #     q[i, 0] = u[i, 0]
        #     for j in range(1, N - 1):
        #         p[i, j] = - f / (d * p[i, j - 1] + e)
        #         q[i, j] = (-a * v[i - 1, j] + (1.0 + 2.0 * a) * v[i, j] -
        #                    c * v[i + 1, j] - d * q[i, j - 1]) / (
        #                        d * p[i, j - 1] + e)
        #     u[i, N - 1] = 1.0
        #     for j in range(N - 2, 0, -1):
        #         u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]
        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = u[1:N - 1, 0]
        for j in range(1, N - 1):
            p[1:N - 1, j] = - f / (d * p[1:N - 1, j - 1] + e)
            q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) *
                             v[1:N - 1, j] - c * v[2:N, j] - d *
                             q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


def init_data(N, datatype):

    u = np.empty((N, N), dtype=datatype)
    v = np.empty((N, N), dtype=datatype)
    p = np.empty((N, N), dtype=datatype)
    q = np.empty((N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            u[i, j] = (i + N - j) / N

    return u, v, p, q


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    TSTEPS, N = 100, 200  # medium dataset
    #TSTEPS, N = 500, 1000 #large dataset
    # 1000, 2000 extra-large dataset
    u, v, p, q = init_data(N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
