# Various deep learning operators and neural networks
import numpy as np

# Size constants
N = 8  #: Batch size
C_in = 3  #: Number of input channels
C_out = 16  #: Number of output features
K = 5  #: Convolution kernel size
H = 32  #: Input height
W = 32  #: Input width


def relu(x):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
def conv2d(input, weights):
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


def conv2d_bias(input, weights, bias):
    return conv2d(input, weights) + bias


# Numerically-stable version of softmax
def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


H_conv1 = H - 4
W_conv1 = W - 4
H_pool1 = H_conv1 // 2
W_pool1 = W_conv1 // 2
H_conv2 = H_pool1 - 4
W_conv2 = W_pool1 - 4
H_pool2 = H_conv2 // 2
W_pool2 = W_conv2 // 2
C_before_fc1 = 16 * H_pool2 * W_pool2


# 2x2 maxpool operator, as used in LeNet-5
def maxpool2d(x):
    output = np.empty(
        [x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]],
        dtype=x.dtype)
    for i in range(x.shape[1] // 2):
        for j in range(x.shape[2] // 2):
            output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2,
                                          2 * j:2 * j + 2, :],
                                        axis=(1, 2))
    return output


# LeNet-5 Convolutional Neural Network (inference mode)
def lenet5(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
           fc3w, fc3b):
    x = relu(conv2d(input, conv1) + conv1bias)
    x = maxpool2d(x)
    x = relu(conv2d(x, conv2) + conv2bias)
    x = maxpool2d(x)
    x = np.reshape(x, (N, C_before_fc1))
    x = relu(x @ fc1w + fc1b)
    x = relu(x @ fc2w + fc2b)
    return x @ fc3w + fc3b


# Batch normalization operator, as used in ResNet
def batchnorm2d(x, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(std + eps)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet_basicblock(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x = relu(x)

    x = conv2d(x, conv2)
    x = batchnorm2d(x)
    x = relu(x)
    x = conv2d(x, conv3)
    x = batchnorm2d(x)
    return relu(x + input)


#############################################################
# Entry points


def test_conv2d():
    # NHWC data layout
    input = np.random.rand(N, H, W, C_in).astype(np.float32)
    # Weights
    weights = np.random.rand(K, K, C_in, C_out).astype(np.float32)
    bias = np.random.rand(C_out).astype(np.float32)

    # Computation
    output = conv2d_bias(input, weights, bias)


def test_softmax():
    # Inputs
    x = np.random.rand(N, C_in, C_out).astype(np.float32)

    # Computation
    output = softmax(x)


def test_mlp():
    mlp_sizes = [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = np.random.rand(C_in, mlp_sizes[0]).astype(np.float32)
    b1 = np.random.rand(mlp_sizes[0]).astype(np.float32)
    w2 = np.random.rand(mlp_sizes[0], mlp_sizes[1]).astype(np.float32)
    b2 = np.random.rand(mlp_sizes[1]).astype(np.float32)
    w3 = np.random.rand(mlp_sizes[1], mlp_sizes[2]).astype(np.float32)
    b3 = np.random.rand(mlp_sizes[2]).astype(np.float32)

    # Computation
    output = mlp(input, w1, b1, w2, b2, w3, b3)


def test_lenet():
    # NHWC data layout
    input = np.random.rand(N, H, W, 1).astype(np.float32)
    # Weights
    conv1 = np.random.rand(5, 5, 1, 6).astype(np.float32)
    conv1_bias = np.random.rand(6).astype(np.float32)
    conv2 = np.random.rand(5, 5, 6, 16).astype(np.float32)
    conv2_bias = np.random.rand(16).astype(np.float32)
    fc1w = np.random.rand(C_before_fc1, 120).astype(np.float32)
    fc1b = np.random.rand(120).astype(np.float32)
    fc2w = np.random.rand(120, 84).astype(np.float32)
    fc2b = np.random.rand(84).astype(np.float32)
    fc3w = np.random.rand(84, 10).astype(np.float32)
    fc3b = np.random.rand(10).astype(np.float32)

    # Computation
    output = lenet5(input, conv1, conv1_bias, conv2, conv2_bias, fc1w, fc1b,
                    fc2w, fc2b, fc3w, fc3b)


def test_resnet_basicblock():
    W = H = 56
    C1 = 256
    C2 = 64
    # Input
    input = np.random.rand(N, H, W, C1).astype(np.float32)
    # Weights
    conv1 = np.random.rand(1, 1, C1, C2).astype(np.float32)
    conv2 = np.random.rand(3, 3, C2, C2).astype(np.float32)
    conv3 = np.random.rand(1, 1, C2, C1).astype(np.float32)

    # Computation
    output = resnet_basicblock(input, conv1, conv2, conv3)


if __name__ == '__main__':
    test_conv2d()
    test_softmax()
    test_mlp()
    test_lenet()
    test_resnet_basicblock()
