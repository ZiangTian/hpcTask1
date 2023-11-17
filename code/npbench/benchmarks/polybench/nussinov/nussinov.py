import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "nussinov"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "molecular_biology",
    dwarf = "dynamic_programming",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "N, seq",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "N, seq",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "N, seq",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "N, gseq",
        setup_str = "gseq = cp.asarray(seq)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "seq=seq, N=N",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "seq=gseq, N=N",
        setup_str = "gseq = cp.asarray(seq)",
        report_str = "DaCe GPU"
    )
)


#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
def match(b1, b2):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


def match_numpy(b1, b2):
    res = np.zeros(b1.shape, dtype=b1.dtype)
    res[b1 + b2 == 3] = 1
    return res


#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)


def kernel_orig(N, seq, table):

    for i in range(N-1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1] +
                                      match(seq[i], seq[j]))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])


def kernel_numpy(N, seq, table):

    # work = np.diag(table, k=1)
    # left = np.diag(table, k=0)
    # lleft = np.diag(table, k=-1)
    # work[0:-1] = np.maximum(np.maximum(work[0:-1], left[0:-2]), left[1:-1])
    # work[0] = np.maximum(work[0], lleft[0])

    # work[1:-1] = np.maximum(work[1:-1], lleft[1:-1] + match_numpy[])
    # work[-1] = max(work[-1], left[-1])

    # for d in range(2, N):
    #     work = np.diag(table, k=d)
    #     left = np.diag(table, k=d-1)
    #     lleft = np.diag(table, k=d-2)
    #     work[0:-1] = np.maximum(np.maximum(work[0:-1], left[0:-2]), left[1:-1])
    #     work[0] = np.maximum(work[0], lleft[])
    #     work[-1] = max(work[-1], left[-1])


    for i in range(N-1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1] +
                                      match(seq[i], seq[j]))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])
            # tmp = np.amax(table[i, i + 1: j], table[i + 2: j + 1, j])
            # table[i, j]


def init_data(N, datatype):

    seq = np.empty((N, ), dtype=datatype)
    table = np.empty((N, N), dtype=datatype)
    for i in range(N):
        seq[i] = (i + 1) % 4
    for i in range(N):
        for j in range(N):
            table[i, j] = 0

    return seq, table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    N = 500  # medium dataset
    # N = 2500  # large dataset
    # N = 5500  # extra-large dataset
    seq, table = init_data(N, np.int32)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
