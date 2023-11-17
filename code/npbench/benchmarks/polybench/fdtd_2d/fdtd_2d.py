import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "fdtd_2d"
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
        arg_str = "TMAX, np_ex, np_ey, np_hz, _fict_",
        setup_str = "np_ex, np_ey, np_hz = np.copy(ex), np.copy(ey), np.copy(hz)",
        report_str = "NumPy",
        out_args = ("np_ex", "np_ey", "np_hz")
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "TMAX, nb_ex, nb_ey, nb_hz, _fict_",
        setup_str = "nb_ex, nb_ey, nb_hz = np.copy(ex), np.copy(ey), np.copy(hz)",
        report_str = "Numba",
        out_args = ("nb_ex", "nb_ey", "nb_hz")
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "TMAX, pt_ex, pt_ey, pt_hz, _fict_",
        setup_str = "pt_ex, pt_ey, pt_hz = np.copy(ex), np.copy(ey), np.copy(hz)",
        report_str = "Pythran",
        out_args = ("pt_ex", "pt_ey", "pt_hz")
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "TMAX, gex, gey, ghz, g_fict_",
        setup_str = "gex, gey, ghz, g_fict_ = cp.asarray(ex), cp.asarray(ey), cp.asarray(hz), cp.asarray(_fict_)",
        report_str = "CuPy",
        out_args = ("gex", "gey", "ghz")
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "ex=dc_ex, ey=dc_ey, hz=dc_hz, _fict_=_fict_, "
                  "TMAX=TMAX, NX=NX, NY=NY",
        setup_str = "dc_ex, dc_ey, dc_hz = np.copy(ex), np.copy(ey), np.copy(hz)",
        report_str = "DaCe CPU",
        out_args = ("dc_ex", "dc_ey", "dc_hz")
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "ex=gex, ey=gey, hz=ghz, _fict_=g_fict_, "
                  "TMAX=TMAX, NX=NX, NY=NY",
        setup_str = "gex, gey, ghz, g_fict_ = cp.asarray(ex), cp.asarray(ey), cp.asarray(hz), cp.asarray(_fict_)",
        report_str = "DaCe GPU",
        out_args = ("gex", "gey", "ghz")
    )
)


def kernel_orig(TMAX, NX, NY, ex, ey, hz, _fict_):
    
    for t in range(TMAX):
        for j in range(NY):
            ey[0, j] = _fict_[t]
        for i in range(1, NX):
            for j in range(NY):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i - 1, j])
        for i in range(NX):
            for j in range(1, NY):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j - 1])
        for i in range(NX - 1):
            for j in range(NY - 1):
                hz[i, j] = hz[i, j] - 0.7 * (ex[i, j + 1] - ex[i, j] +
                                             ey[i + 1, j] - ey[i, j])


def kernel_numpy(TMAX, NX, NY, ex, ey, hz, _fict_):
    
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:NX - 1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:,:NY - 1])
        hz[:NX - 1, :NY - 1] -= 0.7 * (ex[:NX - 1, 1:] - ex[:NX - 1, :NY - 1] +
                                       ey[1:, :NY - 1] - ey[:NX - 1, :NY - 1]) 


def init_data(TMAX, NX, NY, datatype):

    ex = np.empty((NX, NY), dtype=datatype)
    ey = np.empty((NX, NY), dtype=datatype)
    hz = np.empty((NX, NY), dtype=datatype)
    _fict_ = np.empty((TMAX, ), dtype=datatype)
    for i in range(TMAX):
        _fict_[i] = i
    for i in range(NX):
        for j in range(NY):
            ex[i, j] = (i * (j + 1)) / NX
            ey[i, j] = (i * (j + 2)) / NY
            hz[i, j] = (i * (j + 3)) / NX

    return ex, ey, hz, _fict_


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    TMAX, NX, NY = 500, 1000, 1200  # large dataset
    # 1000, 2000, 2600 extra-large dataset
    ex, ey, hz, _fict_ = init_data(TMAX, NX, NY, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
