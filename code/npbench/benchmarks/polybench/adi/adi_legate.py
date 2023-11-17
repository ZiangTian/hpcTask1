import legate.numpy as np
import timeit


def kernel(TSTEPS, N, u):

    v = np.empty(u.shape, dtype=u.dtype)
    p = np.empty(u.shape, dtype=u.dtype)
    q = np.empty(u.shape, dtype=u.dtype)

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = - mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = - mul2 / 2.0
    e = 1.0 + mul2
    f = d
    
    for t in range(1, TSTEPS + 1):
        v[0, 1:N - 1] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = v[0, 1:N - 1]
        for j in range(1, N - 1):
            p[1:N - 1, j] = - c / (a * p[1:N - 1, j - 1] + b)
            q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) *
                             u[j, 1:N - 1] - f * u[j, 2:N] - a *
                             q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
        v[N - 1, 1:N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]

        u[1:N - 1, 0] = 1.0
        p[1:N - 1, 0] = 0.0
        q[1:N - 1, 0] = u[1:N - 1, 0]
        for j in range(1, N - 1):
            p[1:N - 1, j] = - f / (d * p[1:N - 1, j - 1] + e)
            q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) *
                             v[1:N - 1, j] - c * v[2:N, j] - d *
                             q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
        u[1:N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]


def init_data(N, datatype):

    u = np.empty((N, N), dtype=datatype)
    v = np.empty((N, N), dtype=datatype)
    p = np.empty((N, N), dtype=datatype)
    q = np.empty((N, N), dtype=datatype)
    # for i in range(N):
    #     for j in range(N):
    #         u[i, j] = (i + N - j) / N
    u[:] = np.random.randn(N, N)

    return u, v, p, q


if __name__ == "__main__":
    
    # Initialization
    TSTEPS, N = 10, 1000
    u, v, p, q = init_data(N, np.float64)
    lg_u = np.copy(u)

    # First execution
    kernel(TSTEPS, N, lg_u)

    # Benchmark
    time = timeit.repeat("kernel(TSTEPS, N, lg_u)",
                         setup="pass", repeat=20,number=1, globals=globals())
    print("Legate median time: {}".format(np.median(time)))
