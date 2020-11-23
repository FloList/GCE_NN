"""Computes the CDF quantile loss for the discrete dN/dF histograms."""
import tensorflow as tf
# quantile regression:
# adapted from pytorch code here:
# https://github.com/facebookresearch/SingleModelUncertainty/blob/master/aleatoric/regression/joint_estimation.py


def cdf_quantile_loss(p, p_hat, tau, scope="cdf_quantile_loss", weights=None, smoothing=0.0):
    """Compute the CDF quantile loss
    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      tau: quantile for which loss shall be computed
      scope: optional name scope.
    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i
      weights: weight the loss differently for different samples
      smooth: if > 0.0: take the smoothed pinball loss from https://github.com/hatalis/smooth-pinball-neural-network/blob/master/pinball_loss.py
    Returns:
      A 0-D `Tensor` of CDF quantile loss
    """
    with tf.name_scope(scope):
        ecdf_p = tf.math.cumsum(p, axis=-1)
        ecdf_p_hat = tf.math.cumsum(p_hat, axis=-1)
        delta = ecdf_p_hat - ecdf_p

        # Non-smooth C0 loss (default)
        if smoothing == 0.0:
            mask = tf.cast(tf.greater_equal(delta, tf.zeros_like(delta)), tf.float32) - tau
            loss = mask * delta

        # Smooth loss
        else:
            loss = tf.reduce_mean(-tau * delta + smoothing * tf.math.softplus(delta / smoothing))

        if weights is None:
            weights = tf.ones_like(ecdf_p)
        if len(weights.shape) == 1:
            weights = tf.expand_dims(weights, -1)

        weighted_loss = tf.reduce_mean(loss * weights, axis=-1)

        return tf.reduce_mean(weighted_loss)


# Testing
# import numpy as np
# n_test = 10
# truth = tf.expand_dims(tf.random.uniform([n_test]), 0)
# truth /= tf.reduce_sum(truth, 1, keepdims=True)
# est = tf.expand_dims(tf.random.uniform([n_test]), 0)
# est /= tf.reduce_sum(est, 1, keepdims=True)
# # truth = np.zeros((1, n_test))
# # truth[:, 0] = 1
# # truth = tf.constant(truth, dtype=tf.float32)
# # est = np.zeros((1, n_test))
# # est[:, 1] = 1
# # est = tf.constant(est, dtype=tf.float32)
# tau_vec = np.linspace(0.1, 0.9, 9)
# loss_cdf_quantile = [cdf_quantile_loss(truth, est, tau, smoothing=0.001) for tau in tau_vec]
# for i_tau, tau in enumerate(tau_vec):
#     print(np.round(tau, 2), ":", loss_cdf_quantile[i_tau])
# import matplotlib.pyplot as plt
# plt.bar(range(n_test), truth.numpy().flatten(), color="green", alpha=0.5)
# plt.bar(range(n_test), est.numpy().flatten(), color="red", alpha=0.5)
