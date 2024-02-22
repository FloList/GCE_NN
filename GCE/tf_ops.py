import tensorflow as tf
from tensorflow.python.client import device_lib


def get_gpu_names():
    """
    Get list of all GPU devices that Tensorflow can use.
    :return: list of GPU names
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def instance_normalization(x, axis=1, eps=1e-8):
    """
    Instance normalization layer
    :param x: input tensor
    :param axis: axis for normalization (spatial dimension)
    :param eps: small number
    :return: normalized tensor
    """
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + eps)


def normalized_softplus(x, axis=1):
    """
    Normalized softplus activation function
    :param x: input tensor
    :param axis: axis over which the normalization will be computed
    :return: normalized tensor
    """
    return tf.nn.softplus(x) / tf.reduce_sum(tf.nn.softplus(x), axis=axis, keepdims=True)


def split_mean_var(output, mean_act, k, e_bins):
    """
    Format NN output into mean and variances (or in case of Laplace llh: loc. parameter, 2 * scale paramaeter)
    :param output: (batch, k * 2 tensor:
                    k elements: mean, then k elements: variances
    :param mean_act: activation function that will be applied to the mean
    :param k: number of templates
    :return: mean, log variances
    """
    assert k > 0, "k must be positive!"
    assert e_bins > 0, "e_bins must be positive!"
    assert output.shape[1] == e_bins * k * 2, "Aleatoric uncertainty estimation: wrong input shape!"
    # assert output.shape[1] == k * 2, "Aleatoric uncertainty estimation: wrong input shape!"  # for now: no spectra!
    # Activation function should go only over the template dimension
    mean = mean_act(tf.reshape(output[:, :k * e_bins], (-1, k, e_bins)))
    logvar = tf.reshape(output[:, k * e_bins:], (-1, k, e_bins))
    # mean = mean_act(output[:, :k])
    # logvar = output[:, k:]
    return mean, logvar


def split_mean_cov(output, mean_act, k, alpha=0.05, eps=1e-4, norm_const=1.0):
    """
    Format NN output into mean and covariance
    :param output: (batch, k * (k + 3) / 2 tensor:
                   k elements: mean, then k elements: variances, then k * (k - 1) / 2 elements for the correlation
    :param mean_act: activation function that will be applied to the mean (note: for the variances and correlations,
                     exp and tanh are fixed)
    :param k: dimension of regression output
    :param alpha: saturation coeff.
    :param eps: small number for numerical stability
    :param norm_const: scaling of variance variables
    :return: mean, covariance matrix
    """
    assert k > 0, "k must be positive!"
    assert output.shape[1] == (k * (k + 3) // 2), "Covariance estimation: wrong input shape!"
    mean = mean_act(output[:, :k])
    var = tf.math.exp(norm_const * output[:, k:2 * k])
    var_mat = tf.math.sqrt(tf.matmul(tf.expand_dims(var, 2), tf.expand_dims(var, 1)))

    # Now, build correlation matrix
    rhos = (1 - eps) * tf.math.tanh(alpha * output[:, 2 * k:])

    for i_row in range(k):
        if i_row == 0:
            rho_mat_temp = tf.expand_dims(tf.concat([tf.zeros((tf.shape(output)[0], 1)), rhos[:, :k - 1]], axis=1), 1)
        else:
            lower_ind = i_row * k - i_row * (i_row + 1) // 2
            upper_ind = lower_ind + k - 1 - i_row
            new_row = tf.expand_dims(tf.concat([tf.zeros((tf.shape(output)[0], i_row + 1)),
                                                rhos[:, lower_ind:upper_ind]], axis=1), 1)
            rho_mat_temp = tf.concat([rho_mat_temp, new_row], axis=1)

    rho_mat = tf.linalg.band_part(rho_mat_temp, num_lower=0, num_upper=k) \
              + tf.linalg.band_part(tf.linalg.matrix_transpose(rho_mat_temp), num_lower=k, num_upper=0) \
              + tf.tile(tf.expand_dims(tf.eye(k, k), 0), [tf.shape(output)[0], 1, 1])

    return mean, rho_mat * var_mat
