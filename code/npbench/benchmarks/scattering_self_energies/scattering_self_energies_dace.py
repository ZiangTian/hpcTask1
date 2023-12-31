import numpy as np
import dace as dc


NA, NB, Nkz, NE, Nqz, Nw, Norb, N3D = (dc.symbol(s, dc.int64) for s in (
    'NA', 'NB', 'Nkz', 'NE', 'Nqz', 'Nw', 'Norb', 'N3D'))


@dc.program
def scattering_self_energies(neigh_idx: dc.int32[NA, NB],
                             dH: dc.complex128[NA, NB, N3D, Norb, Norb],
                             G: dc.complex128[Nkz, NE, NA, Norb, Norb],
                             D: dc.complex128[Nqz, Nw, NA, NB, N3D, N3D],
                             Sigma: dc.complex128[Nkz, NE, NA, Norb, Norb]):

    # for k in dc.map[0:Nkz]:
    #     for E, q, w, i, j, a, b in dc.map[0:NE, 0:Nqz, 0:Nw, 0:N3D, 0:N3D, 0:NA, 0:NB]:
    # # for k, E, q, w, i, j, a, b in dc.map[0:Nkz, 0:NE, 0:Nqz, 0:Nw, 0:N3D, 0:
    # #                                        N3D, 0:NA, 0:NB]:
    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD
