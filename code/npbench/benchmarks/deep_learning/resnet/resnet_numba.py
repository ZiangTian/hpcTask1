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


# Batch normalization operator, as used in ResNet
@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def batchnorm2d_object(x, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)


# Batch normalization operator, as used in ResNet
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def batchnorm2d_object_parallel(x, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)


# Batch normalization operator, as used in ResNet
@nb.jit(nopython=True, parallel=False, fastmath=True)
def batchnorm2d_nopython(x, eps=1e-5):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.empty(x.shape, dtype=x.dtype)
    mean[:] = np.sum(x, axis=0) / x.shape[0]
    # std = np.std(x, axis=0, keepdims=True)
    std = np.empty(x.shape, dtype=x.dtype)
    std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / x.shape[0])
    return (x - mean) / np.sqrt(std + eps)


# Batch normalization operator, as used in ResNet
@nb.jit(nopython=True, parallel=True, fastmath=True)
def batchnorm2d_nopython_parallel(x, eps=1e-5):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.empty(x.shape, dtype=x.dtype)
    mean[:] = np.sum(x, axis=0) / x.shape[0]
    # std = np.std(x, axis=0, keepdims=True)
    std = np.empty(x.shape, dtype=x.dtype)
    std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / x.shape[0])
    return (x - mean) / np.sqrt(std + eps)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d_object(input, conv1)
    x = batchnorm2d_object(padded)
    x = relu_object_mode(x)

    x = conv2d_object(x, conv2)
    x = batchnorm2d_object(x)
    x = relu_object_mode(x)
    x = conv2d_object(x, conv3)
    x = batchnorm2d_object(x)
    return relu_object_mode(x + input)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_parallel(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d_object_parallel(input, conv1)
    x = batchnorm2d_object_parallel(padded)
    x = relu_object_mode(x)

    x = conv2d_object_parallel(x, conv2)
    x = batchnorm2d_object_parallel(x)
    x = relu_object_mode(x)
    x = conv2d_object_parallel(x, conv3)
    x = batchnorm2d_object_parallel(x)
    return relu_object_mode(x + input)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode_prange(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d_object_prange(input, conv1)
    x = batchnorm2d_object_parallel(padded)
    x = relu_object_mode(x)

    x = conv2d_object_prange(x, conv2)
    x = batchnorm2d_object_parallel(x)
    x = relu_object_mode(x)
    x = conv2d_object_prange(x, conv3)
    x = batchnorm2d_object_parallel(x)
    return relu_object_mode(x + input)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d_nopython(input, conv1)
    x = batchnorm2d_nopython(padded)
    x = relu_nopython_mode(x)

    x = conv2d_nopython(x, conv2)
    x = batchnorm2d_nopython(x)
    x = relu_nopython_mode(x)
    x = conv2d_nopython(x, conv3)
    x = batchnorm2d_nopython(x)
    return relu_nopython_mode(x + input)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d_nopython_parallel(input, conv1)
    x = batchnorm2d_nopython_parallel(padded)
    x = relu_nopython_mode(x)

    x = conv2d_nopython_parallel(x, conv2)
    x = batchnorm2d_nopython_parallel(x)
    x = relu_nopython_mode(x)
    x = conv2d_nopython_parallel(x, conv3)
    x = batchnorm2d_nopython_parallel(x)
    return relu_nopython_mode(x + input)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d_nopython_prange(input, conv1)
    x = batchnorm2d_nopython_parallel(padded)
    x = relu_nopython_mode(x)

    x = conv2d_nopython_prange(x, conv2)
    x = batchnorm2d_nopython_parallel(x)
    x = relu_nopython_mode(x)
    x = conv2d_nopython_prange(x, conv3)
    x = batchnorm2d_nopython_parallel(x)
    return relu_nopython_mode(x + input)
