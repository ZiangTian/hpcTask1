import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "arc_distance"
func_name = "arc_distance"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "kernels",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "t0, p0, t1, p1",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "t0, p0, t1, p1",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "t0, p0, t1, p1",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gt0, gp0, gt1, gp1",
        setup_str = "gt0, gp0, gt1, gp1 = cp.asarray(t0), cp.asarray(p0), "
                    "cp.asarray(t1), cp.asarray(p1)",
        report_str = "Cupy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "theta_1=t0, phi_1=p0, theta_2=t1, phi_2=p1, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gt0, gp0, gt1, gp1",
        setup_str = "gt0, gp0, gt1, gp1 = cp.asarray(t0), cp.asarray(p0), "
                    "cp.asarray(t1), cp.asarray(p1)",
        report_str = "DaCe GPU"
    )
)


def initialize(N):
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    rs = RandomState(MT19937(SeedSequence(123456789)))
    t0, p0, t1, p1 = rs.randn(N), rs.randn(N), rs.randn(N), rs.randn(N)
    return t0, p0, t1, p1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    N = 1000000
    np.random.seed(0)
    t0, p0, t1, p1 = (np.random.randn(N), np.random.randn(N),
                      np.random.randn(N), np.random.randn(N))

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
