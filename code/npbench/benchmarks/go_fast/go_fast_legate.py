import timeit
import legate.numpy as np


def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


if __name__ == "__main__":

    # Initialization
    N = np.int32(10000)
    x = np.arange(N*N).reshape(N, N)

    # First execution
    go_fast(x)

    # Benchmark
    time = timeit.repeat("go_fast(x)", setup="pass", repeat=20,
                         number=1, globals=globals())
    print("Legate median time: {}".format(np.median(time)))
