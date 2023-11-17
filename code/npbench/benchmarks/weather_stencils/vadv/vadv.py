import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "vadv"
func_name = "vadv"
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
        arg_str = "np_utens_stage, u_stage, wcon, u_pos, utens, dtr_stage",
        setup_str = "np_utens_stage = np.copy(utens_stage)",
        report_str = "NumPy",
        out_args = ("np_utens_stage",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "nb_utens_stage, u_stage, wcon, u_pos, utens, dtr_stage",
        setup_str = "nb_utens_stage = np.copy(utens_stage)",
        report_str = "Numba",
        out_args = ("nb_utens_stage",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "pt_utens_stage, u_stage, wcon, u_pos, utens, dtr_stage",
        setup_str = "pt_utens_stage = np.copy(utens_stage)",
        report_str = "Pythran",
        out_args = ("pt_utens_stage",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gutens_stage, gu_stage, gwcon, gu_pos, gutens, dtr_stage",
        setup_str = "gutens_stage, gu_stage, gwcon, gu_pos, gutens = cp.asarray(utens_stage), "
                    "cp.asarray(u_stage), cp.asarray(wcon), cp.asarray(u_pos), cp.asarray(utens)",
        report_str = "CuPy",
        out_args = ("gutens_stage",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "utens_stage=dc_utens_stage, u_stage=u_stage, wcon=wcon, u_pos=u_pos, "
                  "utens=utens, dtr_stage=dtr_stage, I=I, J=J, K=K",
        setup_str = "dc_utens_stage = np.copy(utens_stage)",
        report_str = "DaCe CPU",
        out_args = ("dc_utens_stage",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}".format(func_name),
        arch = "GPU",
        arg_str = "utens_stage=gutens_stage, u_stage=gu_stage, wcon=gwcon, u_pos=gu_pos, "
                  "utens=gutens, dtr_stage=dtr_stage, I=I, J=J, K=K",
        setup_str = "gutens_stage, gu_stage, gwcon, gu_pos, gutens = cp.asarray(utens_stage), "
                    "cp.asarray(u_stage), cp.asarray(wcon), cp.asarray(u_pos), cp.asarray(utens)",
        report_str = "DaCe GPU",
        out_args = ("gutens_stage",)
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
    dtr_stage = 3. / 20.
    I = 256
    J = 256
    K = 160
    # Define arrays
    utens_stage = np.random.rand(I, J, K)
    u_stage = np.random.rand(I, J, K)
    wcon = np.random.rand(I + 1, J, K)
    u_pos = np.random.rand(I, J, K)
    utens = np.random.rand(I, J, K)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())