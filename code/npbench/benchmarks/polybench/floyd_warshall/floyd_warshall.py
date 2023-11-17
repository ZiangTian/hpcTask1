import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "floyd_warshall"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "graph",
    dwarf = "dynamic_programming",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "np_path",
        setup_str = "np_path = np.copy(path)",
        report_str = "NumPy",
        out_args = ("np_path",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "nb_path",
        setup_str = "nb_path = np.copy(path)",
        report_str = "Numba",
        out_args = ("nb_path",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "pt_path",
        setup_str = "pt_path = np.copy(path)",
        report_str = "Pythran",
        out_args = ("pt_path",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gpath",
        setup_str = "gpath = cp.asarray(path)",
        report_str = "CuPy",
        out_args = ("gpath",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "path=dc_path, N=N",
        setup_str = "dc_path = np.copy(path)",
        report_str = "DaCe CPU",
        out_args = ("dc_path",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "path=gpath, N=N",
        setup_str = "gpath = cp.asarray(path)",
        report_str = "DaCe GPU",
        out_args = ("gpath",)
    )
)


def kernel_orig(N, path):

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if path[i, j] >= path[i, k] + path[k, j]:
                    path[i, j] = path[i, k] + path[k, j]


def kernel_numpy(N, path):

    for k in range(N):
        for i in range(N):
            tmp = path[i, k] + path[k, :]
            cond = path[i, :] >= tmp
            path[i, cond] = tmp[cond]


def init_data(N, datatype):

    path = np.empty((N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            path[i, j] = i * j % 7 + 1
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = 999

    return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    #N = 500  # medium dataset
    #N = 1000  # medium+ dataset?
    N = 2800 # large dataset
    # 5600 extra-large dataset
    path = init_data(N, np.int32)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
