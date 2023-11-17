import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "deriche"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "dsp",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, imgIn",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "alpha, imgIn",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha, imgIn",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha, gimgIn",
        setup_str = "gimgIn = cp.asarray(imgIn)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "alpha=alpha, imgIn=imgIn, W=W, H=H",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "alpha=alpha, imgIn=gimgIn, W=W, H=H",
        setup_str = "gimgIn = cp.asarray(imgIn)",
        report_str = "DaCe GPU"
    )
)


def kernel_orig(W, H, alpha, imgIn, imgOut, y1, y2):

    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (
        1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = - k * np.exp(-2.0 * alpha)
    b1 = 2.0 ** (-alpha)
    b2 = - np.exp(-2.0 * alpha)
    c1 = c2 = 1

    for i in range(W):
        ym1 = 0.0
        ym2 = 0.0
        xm1 = 0.0
        for j in range(H):
            y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = imgIn[i, j]
            ym2 = ym1
            ym1 = y1[i, j]
    
    for i in range(W):
        yp1 = 0.0
        yp2 = 0.0
        xp1 = 0.0
        xp2 = 0.0
        for j in range(H-1, -1, -1):
            y2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = imgIn[i, j]
            yp2 = yp1
            yp1 = y2[i, j]
    
    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c1 * (y1[i, j] + y2[i, j])
    
    for j in range(H):
        tm1 = 0.0
        ym1 = 0.0
        ym2 = 0.0
        for i in range(W):
            y1[i, j] = a5 * imgOut[i, j] + a6 * tm1 + b1 * ym1 + b2 * ym2
            tm1 = imgOut[i, j]
            ym2 = ym1
            ym1 = y1[i, j]
    
    for j in range(H):
        tp1 = 0.0
        tp2 = 0.0
        yp1 = 0.0
        yp2 = 0.0
        for i in range(W-1, -1, -1):
            y2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tp2 = tp1
            tp1 = imgOut[i, j]
            yp2 = yp1
            yp1 = y2[i, j]
    
    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c2 * (y1[i, j] + y2[i, j])


def kernel_numpy(W, H, alpha, imgIn, imgOut, y1, y2):

    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (
        1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = - k * np.exp(-2.0 * alpha)
    b1 = 2.0 ** (-alpha)
    b2 = - np.exp(-2.0 * alpha)
    c1 = c2 = 1

    # for i in range(W):
    #     ym1 = 0.0
    #     ym2 = 0.0
    #     xm1 = 0.0
    #     for j in range(H):
    #         y1[i, j] = a1 * imgIn[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
    #         xm1 = imgIn[i, j]
    #         ym2 = ym1
    #         ym1 = y1[i, j]
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, H):
        y1[:, j] = (a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] +
                    b1 * y1[:, j - 1] + b2 * y1[:, j - 2])
    
    # for i in range(W):
    #     yp1 = 0.0
    #     yp2 = 0.0
    #     xp1 = 0.0
    #     xp2 = 0.0
    #     for j in range(H-1, -1, -1):
    #         y2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
    #         xp2 = xp1
    #         xp1 = imgIn[i, j]
    #         yp2 = yp1
    #         yp1 = y2[i, j]
    y2[:, H - 1] = 0.0
    y2[:, H - 2] = a3 * imgIn[:, H - 1]
    # y2[:, H - 2] = a3 * imgIn[:, H - 1] + b1 * y2[:, H - 1]
    for j in range(H - 3, -1, -1):
        y2[:, j] = (a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] +
                    b1 * y2[:, j + 1] + b2 * y2[:, j + 2])
    
    imgOut[:] = c1 * (y1 + y2)
    
    # for j in range(H):
    #     tm1 = 0.0
    #     ym1 = 0.0
    #     ym2 = 0.0
    #     for i in range(W):
    #         y1[i, j] = a5 * imgOut[i, j] + a6 * tm1 + b1 * ym1 + b2 * ym2
    #         tm1 = imgOut[i, j]
    #         ym2 = ym1
    #         ym1 = y1[i, j]
    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, W):
        y1[i, :] = (a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] +
                    b1 * y1[i - 1, :] + b2 * y1[i - 2, :])
    
    # for j in range(H):
    #     tp1 = 0.0
    #     tp2 = 0.0
    #     yp1 = 0.0
    #     yp2 = 0.0
    #     for i in range(W-1, -1, -1):
    #         y2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
    #         tp2 = tp1
    #         tp1 = imgOut[i, j]
    #         yp2 = yp1
    #         yp1 = y2[i, j]
    y2[W - 1, :] = 0.0
    y2[W - 2, :] = a7 * imgOut[W - 1, :]
    # y2[W - 2, :] = a7 * imgOut[W - 1, :] + b1 * y2[W - 1, :]
    for i in range(W - 3, -1, -1):
        y2[i, :] = (a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] +
                    b1 * y2[i + 1, :] + b2 * y2[i + 2, :])
    
    imgOut[:] = c2 * (y1 + y2)


def init_data(W, H, datatype):

    alpha = datatype(0.25)
    imgIn = np.empty((W, H), dtype=datatype)
    imgOut = np.empty((W, H), dtype=datatype)
    y1 = np.empty((W, H), dtype=datatype)
    y2 = np.empty((W, H), dtype=datatype)
    for i in range(W):
        for j in range(H):
            imgIn[i, j] = ((313 * i + 991 * j) % 65536) / 65535.0

    return alpha, imgIn, imgOut, y1, y2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    W, H = 7680, 4320  # extra-large dataset
    alpha, imgIn, imgOut, y1, y2 = init_data(W, H, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
