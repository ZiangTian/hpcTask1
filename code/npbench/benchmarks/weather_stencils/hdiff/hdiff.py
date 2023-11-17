import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "hdiff"
func_name = "hdiff"
domain_name = "climate"
dwarf_name = "structured_grids"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "climate",
    dwarf = "structured_grids",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "in_field, np_out_field, coeff",
        setup_str = "np_out_field = np.copy(out_field)",
        report_str = "NumPy",
        out_args = ("np_out_field",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "in_field, nb_out_field, coeff",
        setup_str = "nb_out_field = np.copy(out_field)",
        report_str = "Numba",
        out_args = ("nb_out_field",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "in_field, pt_out_field, coeff",
        setup_str = "pt_out_field = np.copy(out_field)",
        report_str = "Pythran",
        out_args = ("pt_out_field",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gin_field, gout_field, gcoeff",
        setup_str = "gin_field, gout_field, gcoeff = cp.asarray(in_field), "
                    "cp.asarray(out_field), cp.asarray(coeff)",
        report_str = "CuPy",
        out_args = ("gout_field",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "in_field=in_field, out_field=dc_out_field, coeff=coeff, "
                  "I=I, J=J, K=K",
        setup_str = "dc_out_field = np.copy(out_field)",
        report_str = "DaCe CPU",
        out_args = ("dc_out_field",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}".format(func_name),
        arch = "GPU",
        arg_str = "in_field=gin_field, out_field=gout_field, coeff=gcoeff, "
                  "I=I, J=J, K=K",
        setup_str = "gin_field, gout_field, gcoeff = cp.asarray(in_field), "
                    "cp.asarray(out_field), cp.asarray(coeff)",
        report_str = "DaCe GPU",
        out_args = ("gout_field",)
    )
)


def rng_complex(shape):
    return (np.random.rand(*shape).astype(np.float64) +
            np.random.rand(*shape).astype(np.float64) * 1j)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="dace_cpu")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    I = 256
    J = 256
    K = 160
    # Define arrays
    in_field = np.random.rand(I + 4, J + 4, K)
    out_field = np.random.rand(I, J, K)
    coeff = np.random.rand(I, J, K)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
