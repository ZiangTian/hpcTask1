import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "crc16"
func_name = "crc16"
domain_name = "other"
dwarf_name = "combinatorial"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "other",
    dwarf = "combinatorial",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "data",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "data",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "data",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gdata",
        setup_str = "gdata = cp.asarray(data)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "data=data, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}".format(func_name),
        arch = "GPU",
        arg_str = "data=gdata, N=N",
        setup_str = "gdata = cp.asarray(data)",
        report_str = "DaCe GPU"
    )
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    N = np.int32(1000000)
    data = np.random.randint(0, 256, size=(N,), dtype=np.uint8)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
