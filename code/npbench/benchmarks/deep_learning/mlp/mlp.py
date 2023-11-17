import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "mlp"
func_name = "mlp"
domain_name = "deep_learning"
dwarf_name = "dense_linear_algebra"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "deep_learning",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input, w1, b1, w2, b2, w3, b3",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "input, w1, b1, w2, b2, w3, b3",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input, w1, b1, w2, b2, w3, b3",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "ginput, gw1, gb1, gw2, gb2, gw3, gb3",
        setup_str = "ginput, gw1, gb1, gw2, gb2, gw3, gb3 = cp.asarray(input), "
                    "cp.asarray(w1), cp.asarray(b1), cp.asarray(w2), cp.asarray(b2), "
                    "cp.asarray(w3), cp.asarray(b3)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input=input, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, "
                  "C_in=C_in, N=N, S0=S0, S1=S1, S2=S2",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}_gpu".format(func_name),
        arch = "GPU",
        arg_str = "out=gout, input=ginput, w1=gw1, b1=gb1, w2=gw2, b2=gb2, w3=gw3, b3=gb3, "
                  "C_in=C_in, N=N, S0=S0, S1=S1, S2=S2",
        setup_str = "gout, ginput, gw1, gb1, gw2, gb2, gw3, gb3 = cp.asarray(out), cp.asarray(input), "
                    "cp.asarray(w1), cp.asarray(b1), cp.asarray(w2), cp.asarray(b2), "
                    "cp.asarray(w3), cp.asarray(b3)",
        report_str = "DaCe GPU",
        out_args = ("gout",)
    )
)


def initialize(C_in, C_out, H, K, N, W, S0, S1, S2):
    mlp_sizes = [S0, S1, S2]  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = np.random.rand(C_in, mlp_sizes[0]).astype(np.float32)
    b1 = np.random.rand(mlp_sizes[0]).astype(np.float32)
    w2 = np.random.rand(mlp_sizes[0], mlp_sizes[1]).astype(np.float32)
    b2 = np.random.rand(mlp_sizes[1]).astype(np.float32)
    w3 = np.random.rand(mlp_sizes[1], mlp_sizes[2]).astype(np.float32)
    b3 = np.random.rand(mlp_sizes[2]).astype(np.float32)
    return input, w1, b1, w2, b2, w3, b3


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="dace_gpu")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    # Size constants
    N = 8  #: Batch size
    C_in = 3  #: Number of input channels
    C_out = 16  #: Number of output features
    K = 5  #: Convolution kernel size
    H = 128  #: Input height
    W = 128  #: Input width
    S0, S1, S2 = 30000, 10000, 1000
    input, w1, b1, w2, b2, w3, b3 = initialize(C_in, C_out, H, K, N, W, S0, S1, S2)
    out = np.ndarray((N, S2), input.dtype)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
