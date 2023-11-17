import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "covariance"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "data_mining",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "M, float_n, np_data",
        setup_str = "np_data = np.copy(data)",
        report_str = "NumPy",
        out_args = ("np_data",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "M, float_n, nb_data",
        setup_str = "nb_data = np.copy(data)",
        report_str = "Numba",
        out_args = ("nb_data",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "M, float_n, pt_data",
        setup_str = "pt_data = np.copy(data)",
        report_str = "Pythran",
        out_args = ("pt_data",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "M, float_n, gdata",
        setup_str = "gdata = cp.asarray(data)",
        report_str = "CuPy",
        out_args = ("gdata",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "float_n=float_n, data=dc_data, M=M, N=N",
        setup_str = "dc_data = np.copy(data)",
        report_str = "DaCe CPU",
        out_args = ("dc_data",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "float_n=float_n, data=gdata, M=M, N=N",
        setup_str = "gdata = cp.asarray(data)",
        report_str = "DaCe GPU",
        out_args = ("gdata",)
    )
)


def kernel_orig(M, N, float_n, data):

    mean = np.empty((M,), dtype=data.dtype)
    for j in range(M):
        mean[j] = 0.0
        for i in range(N):
            mean[j] += data[i, j]
        mean[j] /= float_n
  
    for i in range(N):
        for j in range(M):
            data[i, j] -= mean[j]

    cov = np.empty((M, M), dtype=data.dtype)
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = 0.0
            for k in range(N):
                cov[i, j] += data[k, i] * data[k, j]
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

    return cov


def kernel_numpy(M, N, float_n, data):

    # mean = np.sum(data, axis=0) / float_n
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = np.sum(data[:, i] * data[:, j])
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

    return cov


def init_data(M, N, datatype):

    float_n = datatype(N)
    data = np.empty((N, M), dtype=datatype)
    for i in range(N):
        for j in range(M):
            data[i, j] = (i * j) / M

    return float_n, data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    M, N = 1200, 1400  # large dataset
    # 2600, 3000 extra-large dataset
    float_n, data = init_data(M, N, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
