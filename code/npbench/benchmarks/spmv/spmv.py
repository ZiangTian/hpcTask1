import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "spmv"
func_name = "spmv"
domain_name = "kernels"
dwarf_name = "sparse_linear_algebra"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "kernels",
    dwarf = "sparse_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "rows, cols, vals, x",
        setup_str = "pass",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "rows, cols, vals, x",
        setup_str = "pass",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "rows, cols, vals, x",
        setup_str = "pass",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "grows, gcols, gvals, gx",
        setup_str = "grows, gcols, gvals, gx = cp.asarray(rows), "
                    "cp.asarray(cols), cp.asarray(vals), cp.asarray(x)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A_row=rows, A_col=cols, A_val=vals, x=x, M=M, N=N, nnz=nnz",
        setup_str = "pass",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}_gpu".format(func_name),
        arch = "GPU",
        arg_str = "out=gout, A_row=grows, A_col=gcols, A_val=gvals, x=gx, M=M, N=N, nnz=nnz",
        setup_str = "gout, grows, gcols, gvals, gx = cp.asarray(out), cp.asarray(rows), "
                    "cp.asarray(cols), cp.asarray(vals), cp.asarray(x)",
        report_str = "DaCe GPU",
        out_args = ("gout",)
    )
)


def initialize(N, W, H, C1, C2):
    ## Input
    input = np.random.rand(N, H, W, C1).astype(np.float32)
    # Weights
    conv1 = np.random.rand(1, 1, C1, C2).astype(np.float32)
    conv2 = np.random.rand(3, 3, C2, C2).astype(np.float32)
    conv3 = np.random.rand(1, 1, C2, C1).astype(np.float32)
    return (input, conv1, conv2, conv3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="dace_cpu")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    M = N = 8192*16
    nnz = 16384*16

    x = np.random.rand(N)

    # Randomize sparse matrix, assuming uniform sparsity across rows
    rows = np.ndarray(M + 1, dtype=np.uint32)
    cols = np.ndarray(nnz, dtype=np.uint32)
    vals = np.random.rand(nnz)
    nnz_per_row = nnz // M

    # Fill row data
    rows[0] = 0
    rows[1:M] = nnz_per_row
    rows = np.cumsum(rows, dtype=np.uint32)

    # Fill column data
    for i in range(M):
        cols[nnz_per_row*i:nnz_per_row*(i+1)] = \
            np.sort(np.random.choice(N, nnz_per_row, replace=False))
    out = np.ndarray((M, ), vals.dtype)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())


# def test_spmv():
#     M = N = 512
#     nnz = 1024

#     x = np.random.rand(N)

#     # Randomize sparse matrix, assuming uniform sparsity across rows
#     rows = np.ndarray(M + 1, dtype=np.uint32)
#     cols = np.ndarray(nnz, dtype=np.uint32)
#     vals = np.random.rand(nnz)
#     nnz_per_row = nnz // M

#     # Fill row data
#     rows[0] = 0
#     rows[1:M] = nnz_per_row
#     rows = np.cumsum(rows, dtype=np.uint32)

#     # Fill column data
#     for i in range(M):
#         cols[nnz_per_row*i:nnz_per_row*(i+1)] = \
#             np.sort(np.random.choice(N, nnz_per_row, replace=False))

#     output = spmv(rows, cols, vals, x)


# if __name__ == '__main__':
#     test_spmv()
