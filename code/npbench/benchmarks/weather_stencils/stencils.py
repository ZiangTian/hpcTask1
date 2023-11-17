# Stencils based on the COSMO Numerical Weather Prediction model
# Python implementations adapted from GT4Py - GridTools for Python
import numpy as np

# Sample constants
BET_M = 0.5
BET_P = 0.5
dtr_stage = 3. / 20.
I = 128
J = 128
K = 80


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
def vadv(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    ccol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    dcol = np.ndarray((I, J, K), dtype=utens_stage.dtype)
    data_col = np.ndarray((I, J), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, K - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] -
                                  u_stage[:, :, k]) - cs * (
                                      u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (dtr_stage * u_pos[:, :, k] + utens[:, :, k] +
                         utens_stage[:, :, k] + correction_term)

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(K - 1, K - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(K - 2, -1, -1):
        datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


def test_vadv():
    # Define arrays
    utens_stage = np.random.rand(I, J, K)
    u_stage = np.random.rand(I, J, K)
    wcon = np.random.rand(I + 1, J, K)
    u_pos = np.random.rand(I, J, K)
    utens = np.random.rand(I, J, K)

    vadv(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage)


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
def hdiff(in_field, out_field, coeff):
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = np.where(
        (res *
         (in_field[2:I + 3, 2:J + 2, :] - in_field[1:I + 2, 2:J + 2, :])) > 0,
        0,
        res,
    )

    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = np.where(
        (res *
         (in_field[2:I + 2, 2:J + 3, :] - in_field[2:I + 2, 1:J + 2, :])) > 0,
        0,
        res,
    )

    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])


def test_hdiff():
    in_field = np.random.rand(I + 4, J + 4, K)
    out_field = np.random.rand(I, J, K)
    coeff = np.random.rand(I, J, K)
    hdiff(in_field, out_field, coeff)


if __name__ == '__main__':
    test_hdiff()
    test_vadv()
