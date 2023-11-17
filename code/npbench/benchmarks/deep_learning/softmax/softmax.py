import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "softmax"
func_name = "softmax"
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
        arg_str = "x",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "x",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "x",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gx",
        setup_str = "gx = cp.asarray(x)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "x=x, SM=SM, H=H, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}_gpu".format(func_name),
        arch = "GPU",
        arg_str = "x=gx, out=gout, SM=SM, H=H, N=N",
        setup_str = "gx, gout = cp.asarray(x), cp.asarray(out)",
        report_str = "DaCe GPU",
        out_args = ("gout",)
    )
)


def initialize(N, H, SM):
    x = np.random.rand(N, H, SM, SM).astype(np.float32)
    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="dace_cpu")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    # Size constants taken from BERT_large
    N = 64  #: Batch size
    H = 16  #: Number of heads
    SM = 512  #: Sequence length
    x = initialize(N, H, SM)
    out = np.ndarray(x.shape, x.dtype)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
