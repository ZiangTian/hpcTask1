import argparse
import pathlib
import numpy as np
from npbench import run, str2bool


# Module name
module_name = "nbody"
func_name = "nbody"
domain_name = "physics"
dwarf_name = "nbody"

# Framework information
finfo = dict(
    kind = "microapp",
    domain = "physics",
    dwarf = "nbody",
    numpy = dict(
        module_str = "{}_numpy".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "mass, np_pos, np_vel, N, Nt, dt, G, softening",
        setup_str = "np_pos, np_vel = np.copy(pos), np.copy(vel)",
        report_str = "NumPy"
    ),
    numba = dict(
        module_str = "{}_numba".format(module_name),
        func_str = None,  # special names for Numba
        arch = "CPU",
        arg_str = "mass, nb_pos, nb_vel, N, Nt, dt, G, softening",
        setup_str = "nb_pos, nb_vel = np.copy(pos), np.copy(vel)",
        report_str = "Numba"
    ),
    pythran = dict(
        module_str = "{}_pythran".format(module_name),
        module_path = pathlib.Path(__file__).parent.absolute(),
        func_str = func_name,
        arch = "CPU",
        arg_str = "mass, pt_pos, pt_vel, N, Nt, dt, G, softening",
        setup_str = "pt_pos, pt_vel = np.copy(pos), np.copy(vel)",
        report_str = "Pythran"
    ),
    cupy = dict(
        module_str = "{}_cupy".format(module_name),
        func_str = func_name,
        arch = "GPU",
        arg_str = "gmass, gpos, gvel, N, Nt, dt, G, softening",
        setup_str = "gmass, gpos, gvel = cp.asarray(mass), cp.asarray(pos), cp.asarray(vel)",
        report_str = "CuPy"
    ),
    dace_cpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = func_name,
        arch = "CPU",
        arg_str = "mass=mass, pos=dc_pos, vel=dc_vel, N=N, Nt=Nt, dt=dt, G=G, softening=softening",
        setup_str = "dc_pos, dc_vel = np.copy(pos), np.copy(vel)",
        report_str = "DaCe CPU"
    ),
    dace_gpu = dict(
        module_str = "{}_dace".format(module_name),
        func_str = "{}".format(func_name),
        arch = "GPU",
        arg_str = "mass=gmass, pos=gpos, vel=gvel, N=N, Nt=Nt, dt=dt, G=G, softening=softening",
        setup_str = "gmass, gpos, gvel = cp.asarray(mass), cp.asarray(pos), cp.asarray(vel)",
        report_str = "DaCe GPU"
    )
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a", "--append", type=str2bool, nargs="?", default=False)
    args = vars(parser.parse_args())

    # Initialization
    # Simulation parameters
    N = 100    # Number of particles
    t = 0      # current time of the simulation
    tEnd = 10.0   # time at which simulation ends
    dt = 0.01   # timestep
    softening = 0.1    # softening length
    G = 1.0    # Newton's Gravitational Constant
    # Generate Initial Conditions
    np.random.seed(17)            # set the random number generator seed

    mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
    pos  = np.random.randn(N,3)   # randomly selected positions and velocities
    vel  = np.random.randn(N,3)
    np_pos = pos.copy()
    np_vel = vel.copy()
    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())