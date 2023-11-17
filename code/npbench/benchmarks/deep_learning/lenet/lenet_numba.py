import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def relu_object_mode(x):
    return np.maximum(x, 0)


@nb.jit(nopython=True, parallel=False, fastmath=True)
def relu_nopython_mode(x):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def conv2d_object(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# Deep learning convolutional operator (stride = 1)
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def conv2d_object_parallel(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# Deep learning convolutional operator (stride = 1)
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def conv2d_object_prange(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in nb.prange(H_out):
        for j in nb.prange(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i:i + K, j:j + K, :, np.newaxis] *
                weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# Deep learning convolutional operator (stride = 1)
@nb.jit(nopython=True, parallel=False, fastmath=True)
def conv2d_nopython(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            # output[:, i, j, :] = np.sum(
            #     input[:, i:i + K, j:j + K, :, np.newaxis] *
            #     weights[np.newaxis, :, :, :],
            #     axis=(1, 2, 3),
            # )
            # Reshape supported only on contiguous arrays
            inp = input[:, i:i + K, j:j + K, :].copy()
            # Tuple of ints not supported in axis keyword
            output[:, i, j, :] = np.sum(np.sum(np.sum(
                np.reshape(inp, (N, K, K, C_in, 1)) *
                np.reshape(weights, (1, K, K, C_in, C_out)),
                axis=1), axis=1), axis=1
            )

    return output


# Deep learning convolutional operator (stride = 1)
@nb.jit(nopython=True, parallel=True, fastmath=True)
def conv2d_nopython_parallel(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            # output[:, i, j, :] = np.sum(
            #     input[:, i:i + K, j:j + K, :, np.newaxis] *
            #     weights[np.newaxis, :, :, :],
            #     axis=(1, 2, 3),
            # )
            # Reshape supported only on contiguous arrays
            inp = input[:, i:i + K, j:j + K, :].copy()
            # Tuple of ints not supported in axis keyword
            output[:, i, j, :] = np.sum(np.sum(np.sum(
                np.reshape(inp, (N, K, K, C_in, 1)) *
                np.reshape(weights, (1, K, K, C_in, C_out)), axis=1), axis=1),
                axis=1)

    return output


# Deep learning convolutional operator (stride = 1)
@nb.jit(nopython=True, parallel=True, fastmath=True)
def conv2d_nopython_prange(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in nb.prange(H_out):
        for j in nb.prange(W_out):
            # output[:, i, j, :] = np.sum(
            #     input[:, i:i + K, j:j + K, :, np.newaxis] *
            #     weights[np.newaxis, :, :, :],
            #     axis=(1, 2, 3),
            # )
            # Reshape supported only on contiguous arrays
            inp = input[:, i:i + K, j:j + K, :].copy()
            # Tuple of ints not supported in axis keyword
            output[:, i, j, :] = np.sum(np.sum(np.sum(
                np.reshape(inp, (N, K, K, C_in, 1)) *
                np.reshape(weights, (1, K, K, C_in, C_out)), axis=1), axis=1),
                axis=1)

    return output


# 2x2 maxpool operator, as used in LeNet-5
@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def maxpool2d_object(x):
    output = np.empty(
        [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
        dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
                                          2 * j:2 * j + 2, :],
                                        axis=(1, 2))
    return output


# 2x2 maxpool operator, as used in LeNet-5
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def maxpool2d_object_parallel(x):
    output = np.empty(
        [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
        dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
                                          2 * j:2 * j + 2, :],
                                        axis=(1, 2))
    return output


# 2x2 maxpool operator, as used in LeNet-5
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def maxpool2d_object_prange(x):
    output = np.empty(
        [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
        dtype=x.dtype)
    for i in nb.prange(x.shape[1] // 2):
        for j in nb.prange(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
                                          2 * j:2 * j + 2, :],
                                        axis=(1, 2))
    return output


# 2x2 maxpool operator, as used in LeNet-5
@nb.jit(nopython=True, parallel=False, fastmath=True)
def maxpool2d_nopython(x):
    # output = np.empty(
    #     [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
    #     dtype=x.dtype)
    output = np.empty(
        (x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]),
        dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            # output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
            #                               2 * j:2 * j + 2, :],
            #                             axis=(1, 2))
            for k in range(x.shape[0]):
                for l in range(x.shape[3]):
                    output[k, i, j, l] = np.max(
                        x[k, 2 * i:2 * i + 2, 2 * j:2 * j + 2, l])
    return output


# 2x2 maxpool operator, as used in LeNet-5
@nb.jit(nopython=True, parallel=True, fastmath=True)
def maxpool2d_nopython_parallel(x):
    # output = np.empty(
    #     [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
    #     dtype=x.dtype)
    output = np.empty(
        (x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]),
        dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            # output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
            #                               2 * j:2 * j + 2, :],
            #                             axis=(1, 2))
            for k in range(x.shape[0]):
                for l in range(x.shape[3]):
                    output[k, i, j, l] = np.max(
                        x[k, 2 * i:2 * i + 2, 2 * j:2 * j + 2, l])
    return output


# 2x2 maxpool operator, as used in LeNet-5
@nb.jit(nopython=True, parallel=True, fastmath=True)
def maxpool2d_nopython_prange(x):
    # output = np.empty(
    #     [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
    #     dtype=x.dtype)
    output = np.empty(
        (x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]),
        dtype=x.dtype)
    for i in nb.prange(x.shape[1] // 2):
        for j in nb.prange(x.shape[2] // 2):
            # output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
            #                               2 * j:2 * j + 2, :],
            #                             axis=(1, 2))
            for k in nb.prange(x.shape[0]):
                for l in nb.prange(x.shape[3]):
                    output[k, i, j, l] = np.max(
                        x[k, 2 * i:2 * i + 2, 2 * j:2 * j + 2, l])
    return output


# LeNet-5 Convolutional Neural Network (inference mode)
@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w,
                fc2b, fc3w, fc3b, N, C_before_fc1):
    x = relu_object_mode(conv2d_object(input, conv1) + conv1bias)
    x = maxpool2d_object(x)
    x = relu_object_mode(conv2d_object(x, conv2) + conv2bias)
    x = maxpool2d_object(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_object_mode(x @ fc1w + fc1b)
    x = relu_object_mode(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


# LeNet-5 Convolutional Neural Network (inference mode)
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b,
                         fc2w, fc2b, fc3w, fc3b, N, C_before_fc1):
    x = relu_object_mode(conv2d_object_parallel(input, conv1) + conv1bias)
    x = maxpool2d_object_parallel(x)
    x = relu_object_mode(conv2d_object_parallel(x, conv2) + conv2bias)
    x = maxpool2d_object_parallel(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_object_mode(x @ fc1w + fc1b)
    x = relu_object_mode(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


# LeNet-5 Convolutional Neural Network (inference mode)
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_prange(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b,
                       fc2w, fc2b, fc3w, fc3b, N, C_before_fc1):
    x = relu_object_mode(conv2d_object_prange(input, conv1) + conv1bias)
    x = maxpool2d_object_prange(x)
    x = relu_object_mode(conv2d_object_prange(x, conv2) + conv2bias)
    x = maxpool2d_object_prange(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_object_mode(x @ fc1w + fc1b)
    x = relu_object_mode(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


# LeNet-5 Convolutional Neural Network (inference mode)
@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w,
                  fc2b, fc3w, fc3b, N, C_before_fc1):
    x = relu_nopython_mode(conv2d_nopython(input, conv1) + conv1bias)
    x = maxpool2d_nopython(x)
    x = relu_nopython_mode(conv2d_nopython(x, conv2) + conv2bias)
    x = maxpool2d_nopython(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_nopython_mode(x @ fc1w + fc1b)
    x = relu_nopython_mode(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


# LeNet-5 Convolutional Neural Network (inference mode)
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(input, conv1, conv1bias, conv2, conv2bias, fc1w,
                           fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1):
    x = relu_nopython_mode(conv2d_nopython_parallel(input, conv1) + conv1bias)
    x = maxpool2d_nopython_parallel(x)
    x = relu_nopython_mode(conv2d_nopython_parallel(x, conv2) + conv2bias)
    x = maxpool2d_nopython_parallel(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_nopython_mode(x @ fc1w + fc1b)
    x = relu_nopython_mode(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


# LeNet-5 Convolutional Neural Network (inference mode)
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(input, conv1, conv1bias, conv2, conv2bias, fc1w,
                         fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1):
    x = relu_nopython_mode(conv2d_nopython_prange(input, conv1) + conv1bias)
    x = maxpool2d_nopython_prange(x)
    x = relu_nopython_mode(conv2d_nopython_prange(x, conv2) + conv2bias)
    x = maxpool2d_nopython_prange(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu_nopython_mode(x @ fc1w + fc1b)
    x = relu_nopython_mode(x @ fc2w + fc2b)
    return x @ fc3w + fc3b
