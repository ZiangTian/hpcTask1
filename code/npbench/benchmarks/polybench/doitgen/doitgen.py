import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "doitgen"
func_name = "kernel"

# Framework information
finfo = dict(
    kind = "microbench",
    domain = "kernels",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "NR, NQ, NP, np_A, C4",
        setup_str = "np_A = np.copy(A)",
        report_str = "NumPy",
        out_args = ("np_A",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "NR, NQ, NP, nb_A, C4",
        setup_str = "nb_A = np.copy(A)",
        report_str = "Numba",
        out_args = ("nb_A",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "NR, NQ, NP, pt_A, C4",
        setup_str = "pt_A = np.copy(A)",
        report_str = "Pythran",
        out_args = ("pt_A",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "NR, NQ, NP, gA, gC4",
        setup_str = "gA, gC4 = cp.asarray(A), cp.asarray(C4)",
        report_str = "CuPy",
        out_args = ("gA",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "A=dc_A, C4=C4, NR=NR, NQ=NQ, NP=NP",
        setup_str = "dc_A = np.copy(A)",
        report_str = "DaCe CPU",
        out_args = ("dc_A",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "A=gA, C4=gC4, NR=NR, NQ=NQ, NP=NP",
        setup_str = "gA, gC4 = cp.asarray(A), cp.asarray(C4)",
        report_str = "DaCe GPU",
        out_args = ("gA",)
    )
)


def kernel_orig(NR, NQ, NP, A, C4, sum):

    for r in range(NR):
        for q in range(NQ):
            for p in range(NP):
                sum[p] = 0.0
                for s in range(NP):
                    sum[p] += A[r, q, s] * C4[s, p]
            for p in range(NP):
                A[r, q, p] = sum[p]


def kernel_numpy(NR, NQ, NP, A, C4, sum):

    for r in range(NR):
        for q in range(NQ):
            sum[:] = A[r, q, :] @ C4
            A[r, q, :] = sum
    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


def init_data(NR, NQ, NP, datatype):

    A = np.empty((NR, NQ, NP), dtype=datatype)
    C4 = np.empty((NP, NP, ), dtype=datatype)
    sum = np.empty((NP, ), dtype=datatype)
    for i in range(NR):
        for j in range(NQ):
            for k in range(NP):
                A[i, j, k] = ((i * j + k) % NP) / NP
    for i in range(NP):
        for j in range(NP):
            C4[i, j] = (i * j % NP) / NP 

    return A, C4, sum


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())
    
    # Initialization
    NR, NQ, NP = 220, 250, 270  # extra-large dataset
    A, C4, sum = init_data(NR, NQ, NP, np.float64)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())

    # np_A = np.copy(A)
    # nb_A = np.copy(A)
    # dc_A = np.copy(A)
    # dc_exec, _ = benchmark("dc_impl.kernel.compile()",
    #                        out_text="DaCe compilation time", context=globals())

    # # First execution
    # benchmark("np_impl.kernel(NR, NQ, NP, np_A, C4)",
    #           out_text="NumPy first execution", context=globals())
    # benchmark("nb_impl.kernel(NR, NQ, NP, nb_A, C4)",
    #           out_text="Numba first execution", context=globals())
    # benchmark("dc_exec(A=dc_A, C4=C4, NR=NR, NQ=NQ, NP=NP)",
    #           out_text="DaCe first execution", context=globals())
    # # np_impl.kernel(NR, NQ, NP, np_A, C4)
    # # nb_impl.kernel(NR, NQ, NP, nb_A, C4)
    # # dace_exec(A=dc_A, C4=C4, NR=NR, NQ=NQ, NP=NP)

    # # Validation
    # validation(np_A, nb_A, framework="Numba")
    # validation(np_A, dc_A, framework="DaCe")
    # # assert(np.allclose(np_A, nb_A))
    # # assert(np.allclose(np_A, dc_A))

    # # Benchmark
    # benchmark("np_impl.kernel(NR, NQ, NP, np_A, C4)",
    #           setup="np_A = np.copy(A)",
    #           out_text="NumPy median time",
    #           repeat=10, context=globals())
    # benchmark("nb_impl.kernel(NR, NQ, NP, nb_A, C4)",
    #           setup="nb_A = np.copy(A)",
    #           out_text="Numba median time",
    #           repeat=10, context=globals())
    # benchmark("dc_exec(A=dc_A, C4=C4, NR=NR, NQ=NQ, NP=NP)",
    #           setup="dc_A = np.copy(A)",
    #           out_text="DaCe median time",
    #           repeat=10, context=globals())
    # # time = timeit.repeat("np_impl.kernel(NR, NQ, NP, np_A, C4)",
    # #                      setup="np_A = np.copy(A)",
    # #                      repeat=20, number=1, globals=globals())
    # # print("Numpy median time: {}".format(np.median(time)))
    # # time = timeit.repeat("nb_impl.kernel(NR, NQ, NP, nb_A, C4)",
    # #                       setup="nb_A = np.copy(A)",
    # #                       repeat=20, number=1, globals=globals())
    # # print("Numba median time: {}".format(np.median(time)))
    # # time = timeit.repeat("dace_exec(A=dc_A, C4=C4, NR=NR, NQ=NQ, NP=NP)",
    # #                      setup="dc_A = np.copy(A)",
    # #                      repeat=20, number=1, globals=globals())
    # # print("DaCe median time: {}".format(np.median(time)))
