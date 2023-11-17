import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "conv2d"
func_name = "conv2d_bias"
domain_name = "deep_learning"
dwarf_name = "dense_linear_algebra"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "deep_learning",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input, weights, bias",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "input, weights, bias",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input, weights, bias",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "ginput, gweights, gbias",
        setup_str = "ginput, gweights, gbias = cp.asarray(input), "
                    "cp.asarray(weights), cp.asarray(bias)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input=input, weights=weights, bias=bias, "
                  "C_in=C_in, C_out=C_out, H=H, K=K, N=N, W=W",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "input=ginput, weights=gweights, bias=gbias, "
                  "C_in=C_in, C_out=C_out, H=H, K=K, N=N, W=W",
        setup_str = "ginput, gweights, gbias = cp.asarray(input), "
                    "cp.asarray(weights), cp.asarray(bias)",
        report_str = "DaCe GPU"
    )
)


def initialize(C_in, C_out, H, K, N, W):
    # NHWC data layout
    input = np.random.rand(N, H, W, C_in).astype(np.float32)
    # Weights
    weights = np.random.rand(K, K, C_in, C_out).astype(np.float32)
    bias = np.random.rand(C_out).astype(np.float32)
    return input, weights, bias


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
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
    K = 20  #: Convolution kernel size
    H = 256  #: Input height
    W = 256  #: Input width
    input, weights, bias = initialize(C_in, C_out, H, K, N, W)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
