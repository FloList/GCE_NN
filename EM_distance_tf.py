"""Contains the EM distance, and also the cumulative Jensen-Shannon divergence."""
import tensorflow as tf
# EM2: adapted from https://github.com/master/nima/blob/master/nima.py#L58 and massively simplified (simply use cumsum)
# CJS: adapted from https://github.com/luke321321/portfolio/blob/master/climbing/CNN.ipynb


def emd_loss(p, p_hat, r=2, scope="emd_loss", weights=None, do_root=False):
    """Compute the Earth Mover's Distance loss.
    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
    arXiv:1611.05916 (2016).
    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      r: a constant for the r-norm.
      scope: optional name scope.
      do_root: if True: raise result to the power of "1/r"
    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i
      weights: weight the loss differently for different samples
    Returns:
      A 0-D `Tensor` of r-normed EMD loss.
    """
    with tf.name_scope(scope):
        ecdf_p = tf.math.cumsum(p, axis=-1)
        ecdf_p_hat = tf.math.cumsum(p_hat, axis=-1)
        if weights is None:
            weights = tf.ones_like(ecdf_p)
        if len(weights.shape) == 1:
            weights = tf.expand_dims(weights, -1)
        if r == 1:
            emd = tf.reduce_mean(tf.abs(ecdf_p - ecdf_p_hat) * weights, axis=-1)
        elif r == 2:
            emd = tf.reduce_mean((ecdf_p - ecdf_p_hat) ** 2 * weights, axis=-1)
            if do_root:
                emd = tf.sqrt(emd)
        else:
            emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat) * weights, r), axis=-1)
            if do_root:
                emd = tf.pow(emd, 1 / r)
        return tf.reduce_mean(emd)




def cjs_loss(p, p_hat, scope="cjs_loss", weights=None, eps=1e-10):
    """Computes the symmetrical discrete cumulative Jensen-Shannon divergence from https://arxiv.org/pdf/1708.07089.pdf

    Inputs:
    - labels: Tensor of shape [batch_size] and dtype int32 or int64.
      Each entry in labels must be an index in [0, num_classes)
    - logits: Unscaled log probabilities of shape [batch_size, num_classes]

    Returns:
    - loss: A Tensor of the same shape as labels and of the same type as logits with the softmax cross entropy loss.
    """
    with tf.name_scope(scope):
        cdf_labels = tf.cumsum(p, axis=-1)
        cdf_logits = tf.cumsum(p_hat, axis=-1)

        def ACCJS(p_, q_):
            with tf.name_scope("ACCJS"):
            # if p(i) = 0 then ACCJS(p, q)(i) = 0 since xlog(x) -> 0 as x-> 0
                p_ = tf.clip_by_value(p_, eps, 1.0)
                return 0.5 * tf.reduce_sum(p_ * tf.math.log(p_ / (0.5 * (p_ + q_))), axis=-1)

        loss = ACCJS(cdf_logits, cdf_labels) + ACCJS(cdf_labels, cdf_logits)
        weights = tf.reshape(weights, [-1])
        return tf.reduce_mean(loss * weights)


# Testing
# import numpy as np
# n_test = 10
# # truth = tf.expand_dims(tf.random.uniform([n_test]), 0)
# # truth /= tf.reduce_sum(truth, 1, keepdims=True)
# # est = tf.expand_dims(tf.random.uniform([n_test]), 0)
# # est /= tf.reduce_sum(est, 1, keepdims=True)
# truth = np.zeros((1, n_test))
# truth[:, 0] = 1
# truth = tf.constant(truth, dtype=tf.float32)
# est = np.zeros((1, n_test))
# est[:, 1] = 1
# est = tf.constant(est, dtype=tf.float32)
# loss_emd = emd_loss(truth, est, r=1, scope="EmdLoss")
# loss_cjs = cjs_loss(truth, est)
# import matplotlib.pyplot as plt
# plt.bar(range(n_test), truth.numpy().flatten(), color="green", alpha=0.5)
# plt.bar(range(n_test), est.numpy().flatten(), color="red", alpha=0.5)
