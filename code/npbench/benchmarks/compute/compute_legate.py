import timeit
import legate.numpy as np


def compute(array_1, array_2, a, b, c):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c


if __name__ == "__main__":
    
    # Initialization
    M = np.intc(10000)
    N = np.intc(10000)
    array_1 = np.random.uniform(0, 1000, size=(M, N)).astype(np.intc)
    array_2 = np.random.uniform(0, 1000, size=(M, N)).astype(np.intc)
    a = np.intc(4)
    b = np.intc(3)
    c = np.intc(9)

    # First execution
    compute(array_1, array_2, a, b, c)

    # Benchmark
    time = timeit.repeat("compute(array_1, array_2, a, b, c)",
                         setup="pass", repeat=20, number=1, globals=globals())
    print("Legate median time: {}".format(np.median(time)))

