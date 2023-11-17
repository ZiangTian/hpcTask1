import argparse
import pathlib
import numpy as np
from npbench import run, run_dace, run_numba, str2bool


# Module name
module_name = "scattering_self_energies"
func_name = "scattering_self_energies"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "qts",
    dwarf = "dense_linear_algebra",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "neigh_idx, dH, G, D, np_Sigma",
        setup_str = "np_Sigma = np.copy(Sigma)",
        report_str = "NumPy",
        out_args = ("np_Sigma",)
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "neigh_idx, dH, G, D, nb_Sigma",
        setup_str = "nb_Sigma = np.copy(Sigma)",
        report_str = "Numba",
        out_args = ("nb_Sigma",)
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "neigh_idx, dH, G, D, pt_Sigma",
        setup_str = "pt_Sigma = np.copy(Sigma)",
        report_str = "Pythran",
        out_args = ("pt_Sigma",)
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gneigh_idx, gdH, gG, gD, gSigma",
        setup_str = "gneigh_idx, gdH, gG, gD, gSigma = cp.asarray(neigh_idx), cp.asarray(dH), "
                    "cp.asarray(G), cp.asarray(D), cp.asarray(Sigma)",
        report_str = "Cupy",
        out_args = ("gSigma",)
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "neigh_idx=neigh_idx, dH=dH, G=G, D=D, Sigma=dc_Sigma, NA=NA, "
                  "NB=NB, Nkz=Nkz, NE=NE, Nqz=Nqz, Nw=Nw, N3D=N3D, Norb=Norb",
        setup_str = "dc_Sigma = np.copy(Sigma)",
        report_str = "DaCe CPU",
        out_args = ("dc_Sigma",)
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "neigh_idx=gneigh_idx, dH=gdH, G=gG, D=gD, Sigma=gSigma, NA=NA, "
                  "NB=NB, Nkz=Nkz, NE=NE, Nqz=Nqz, Nw=Nw, N3D=N3D, Norb=Norb",
        setup_str = "gneigh_idx, gdH, gG, gD, gSigma = cp.asarray(neigh_idx), cp.asarray(dH), "
                    "cp.asarray(G), cp.asarray(D), cp.asarray(Sigma)",
        report_str = "DaCe GPU",
        out_args = ("gSigma",)
    )
)


def rng_complex(shape):
    return (np.random.rand(*shape).astype(np.float64) +
            np.random.rand(*shape).astype(np.float64) * 1j)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    Nkz = 4
    NE = 10
    Nqz = 4
    Nw = 3
    N3D = 3
    NA = 20
    NB = 4
    Norb = 4
    neigh_idx = np.ndarray([NA, NB], dtype=np.int32)
    for i in range(NA):
        neigh_idx[i] = np.positive(np.arange(i-NB/2, i+NB/2) % NA)
    dH = rng_complex([NA, NB, N3D, Norb, Norb])
    G = rng_complex([Nkz, NE, NA, Norb, Norb])
    D = rng_complex([Nqz, Nw, NA, NB, N3D, N3D])
    Sigma = np.zeros([Nkz, NE, NA, Norb, Norb], dtype=np.complex128)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())

    # nb_S = np.zeros([Nkz, NE, NA, Norb, Norb], dtype=np.complex128)
    # dc_S = np.zeros([Nkz, NE, NA, Norb, Norb], dtype=np.complex128)
    # dace_exec = dc_impl.scattering_self_energies.compile()

    # # First execution
    # np_impl.scattering_self_energies(neigh_idx, dH, G, D, np_S)
    # nb_impl.scattering_self_energies(neigh_idx, dH, G, D, nb_S)
    # # dace_exec(neigh_idx=neigh_idx, dH=dH, G=G, D=D, Sigma=dc_S,
    # #           NA=NA, NB=NB, Nkz=Nkz, NE=NE, Nqz=Nqz, Nw=Nw, N3D=N3D, Norb=Norb)

    # # Benchmark
    # time = timeit.repeat(
    #     "np_impl.scattering_self_energies(neigh_idx, dH, G, D, np_S)",
    #     setup="pass", repeat=10, number=1, globals=globals())
    # print("NumPy Median time: {}".format(np.median(time)))
    # time = timeit.repeat(
    #     "nb_impl.scattering_self_energies(neigh_idx, dH, G, D, nb_S)",
    #     setup="pass", repeat=10, number=1, globals=globals())
    # print("Numba Median time: {}".format(np.median(time)))
    # time = timeit.repeat(
    #     "dace_exec(neigh_idx=neigh_idx, dH=dH, G=G, D=D, Sigma=dc_S, "
    #     "NA=NA, NB=NB, Nkz=Nkz, NE=NE, Nqz=Nqz, Nw=Nw, N3D=N3D, Norb=Norb)",
    #     setup="pass", repeat=10, number=1, globals=globals())
    # print("DaCe Median time: {}".format(np.median(time)))

    # # Validation
    # assert(np.allclose(np_S, nb_S))
    # assert(np.allclose(np_S, dc_S))
