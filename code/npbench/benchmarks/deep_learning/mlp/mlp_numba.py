import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def relu_object_mode(x):
    return np.maximum(x, 0)


@nb.jit(nopython=True, parallel=False, fastmath=True)
def relu_nopython_mode(x):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def softmax_object_mode(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# Numerically-stable version of softmax
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def softmax_object_mode_parallel(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# Numerically-stable version of softmax
@nb.jit(nopython=True, parallel=False, fastmath=True)
def softmax_nopython_mode(x):
    new_shape = (x.shape[0], 1)
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.empty(new_shape, dtype=x.dtype)
    for i in range(x.shape[1]):
        tmp_max[:, 0] = np.max(x[:, i])
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.reshape(np.sum(tmp_out, axis=-1), new_shape)
    return tmp_out / tmp_sum


# Numerically-stable version of softmax
@nb.jit(nopython=True, parallel=True, fastmath=True)
def softmax_nopython_mode_parallel(x):
    new_shape = (x.shape[0], 1)
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.empty(new_shape, dtype=x.dtype)
    for i in range(x.shape[1]):
        tmp_max[:, 0] = np.max(x[:, i])
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.reshape(np.sum(tmp_out, axis=-1), new_shape)
    return tmp_out / tmp_sum


# Numerically-stable version of softmax
@nb.jit(nopython=True, parallel=True, fastmath=True)
def softmax_nopython_mode_prange(x):
    new_shape = (x.shape[0], 1)
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.empty(new_shape, dtype=x.dtype)
    for i in nb.prange(x.shape[1]):
        tmp_max[:, 0] = np.max(x[:, i])
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.reshape(np.sum(tmp_out, axis=-1), new_shape)
    return tmp_out / tmp_sum


# 3-layer MLP
@nb.jit(nopython=False, forceobj=True, parallel=False, fastmath=True)
def object_mode(input, w1, b1, w2, b2, w3, b3):
    x = relu_object_mode(input @ w1 + b1)
    x = relu_object_mode(x @ w2 + b2)
    x = softmax_object_mode(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


# 3-layer MLP
@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def object_mode(input, w1, b1, w2, b2, w3, b3):
    x = relu_object_mode(input @ w1 + b1)
    x = relu_object_mode(x @ w2 + b2)
    x = softmax_object_mode_parallel(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


# 3-layer MLP
@nb.jit(nopython=True, parallel=False, fastmath=True)
def nopython_mode(input, w1, b1, w2, b2, w3, b3):
    x = relu_nopython_mode(input @ w1 + b1)
    x = relu_nopython_mode(x @ w2 + b2)
    x = softmax_nopython_mode(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


# 3-layer MLP
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_parallel(input, w1, b1, w2, b2, w3, b3):
    x = relu_nopython_mode(input @ w1 + b1)
    x = relu_nopython_mode(x @ w2 + b2)
    x = softmax_nopython_mode_parallel(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


# 3-layer MLP
@nb.jit(nopython=True, parallel=True, fastmath=True)
def nopython_mode_prange(input, w1, b1, w2, b2, w3, b3):
    x = relu_nopython_mode(input @ w1 + b1)
    x = relu_nopython_mode(x @ w2 + b2)
    x = softmax_nopython_mode_prange(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x
