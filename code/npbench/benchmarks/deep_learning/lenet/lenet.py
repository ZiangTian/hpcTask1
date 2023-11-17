import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "lenet"
func_name = "lenet5"
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
        arg_str = "input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, "
                  "fc2b, fc3w, fc3b, N, C_before_fc1",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, "
                  "fc2b, fc3w, fc3b, N, C_before_fc1",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, "
                  "fc2b, fc3w, fc3b, N, C_before_fc1",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "ginput, gconv1, gconv1bias, gconv2, gconv2bias, gfc1w, gfc1b, gfc2w, "
                  "gfc2b, gfc3w, gfc3b, N, C_before_fc1",
        setup_str = "ginput, gconv1, gconv1bias, gconv2, gconv2bias, gfc1w, gfc1b, gfc2w, "
                    "gfc2b, gfc3w, gfc3b = cp.asarray(input), cp.asarray(conv1), cp.asarray(conv1bias), "
                    "cp.asarray(conv2), cp.asarray(conv2bias), cp.asarray(fc1w), cp.asarray(fc1b), "
                    "cp.asarray(fc2w), cp.asarray(fc2b), cp.asarray(fc3w), cp.asarray(fc3b)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "input=input, conv1=conv1, conv1bias=conv1bias, conv2=conv2, "
                  "conv2bias=conv2bias, fc1w=fc1w, fc1b=fc1b, fc2w=fc2w, "
                  "fc2b=fc2b, fc3w=fc3w, fc3b=fc3b, N=N, H=H, W=W, "
                  "C_before_fc1=C_before_fc1",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}_gpu".format(func_name),
        arch = "GPU",
        arg_str = "out=gout, input=ginput, conv1=gconv1, conv1bias=gconv1bias, conv2=gconv2, "
                  "conv2bias=gconv2bias, fc1w=gfc1w, fc1b=gfc1b, fc2w=gfc2w, "
                  "fc2b=gfc2b, fc3w=gfc3w, fc3b=gfc3b, N=N, H=H, W=W, "
                  "C_before_fc1=C_before_fc1",
        setup_str = "gout, ginput, gconv1, gconv1bias, gconv2, gconv2bias, gfc1w, gfc1b, gfc2w, "
                    "gfc2b, gfc3w, gfc3b = cp.asarray(out), cp.asarray(input), cp.asarray(conv1), cp.asarray(conv1bias), "
                    "cp.asarray(conv2), cp.asarray(conv2bias), cp.asarray(fc1w), cp.asarray(fc1b), "
                    "cp.asarray(fc2w), cp.asarray(fc2b), cp.asarray(fc3w), cp.asarray(fc3b)",
        report_str = "DaCe GPU",
        out_args = ("gout",)
    )
)


def initialize(C_in, C_out, H, K, N, W, C_before_fc1):
    # NHWC data layout
    input = np.random.rand(N, H, W, 1).astype(np.float32)
    # Weights
    conv1 = np.random.rand(5, 5, 1, 6).astype(np.float32)
    conv1bias = np.random.rand(6).astype(np.float32)
    conv2 = np.random.rand(5, 5, 6, 16).astype(np.float32)
    conv2bias = np.random.rand(16).astype(np.float32)
    fc1w = np.random.rand(C_before_fc1, 120).astype(np.float32)
    fc1b = np.random.rand(120).astype(np.float32)
    fc2w = np.random.rand(120, 84).astype(np.float32)
    fc2b = np.random.rand(84).astype(np.float32)
    fc3w = np.random.rand(84, 10).astype(np.float32)
    fc3b = np.random.rand(10).astype(np.float32)
    return (input, conv1, conv1bias, conv2, conv2bias,
            fc1w, fc1b, fc2w, fc2b, fc3w, fc3b)


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
    N = 16  #: Batch size
    C_in = 3  #: Number of input channels
    C_out = 16  #: Number of output features
    K = 5  #: Convolution kernel size
    H = 256  #: Input height
    W = 256  #: Input width
    H_conv1 = H - 4
    W_conv1 = W - 4
    H_pool1 = H_conv1 // 2
    W_pool1 = W_conv1 // 2
    H_conv2 = H_pool1 - 4
    W_conv2 = W_pool1 - 4
    H_pool2 = H_conv2 // 2
    W_pool2 = W_conv2 // 2
    C_before_fc1 = 16 * H_pool2 * W_pool2
    (input, conv1, conv1bias, conv2, conv2bias,
     fc1w, fc1b, fc2w, fc2b, fc3w, fc3b) = initialize(
         C_in, C_out, H, K, N, W, C_before_fc1)
    out = np.ndarray((N, 10), input.dtype)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
