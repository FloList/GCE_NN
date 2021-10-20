import tensorflow as tf


def get_loss_and_keys(which, params, ff_means_only=False):
    """
    Returns the loss function.
    :param which: submodel, must be one of "flux_fractions" or "histograms"
    :param params: parameter dictionary
    :param ff_means_only: even if aleatoric uncertainties for flux fractions are enabled, only train the means
    :return: loss function, list of keys required (apart from true label, which is always assumed to be first input)
    """
    if which == "flux_fractions":
        loss, loss_keys = get_loss_and_keys_flux_fractions(params.train["ff_loss"],
                                                           do_var=params.nn.ff["alea_var"] and not ff_means_only,
                                                           do_covar=params.nn.ff["alea_covar"] and not ff_means_only)
    elif which == "histograms":
        loss, loss_keys = get_loss_and_keys_histograms(params.train["hist_loss"],
                                                       smoothing_empl=params.train["hist_pinball_smoothing"])
    else:
        raise NotImplementedError
    return loss, loss_keys


def get_loss_and_keys_flux_fractions(ff_loss_str, do_var=False, do_covar=False):
    """
    Returns the loss function for the flux fraction estimation.
    :param ff_loss_str: : string specifying histogram loss
    :param do_var: estimate aleatoric variances?
    :param do_covar: estimate aleatoric covariance matrix?
    :return: loss function, list of keys required (apart from true label, which is always assumed to be first input)
    """
    assert not (do_var and do_covar), "Either 'do_var' or 'do_covar' should be chosen, not both!"
    if do_var or do_covar:
        assert ff_loss_str.lower() in ["l2", "mse"], "For flux fraction uncertainty estimation choose 'l2' loss!"

    if ff_loss_str.lower() in ["l2", "mse"]:
        if do_covar:
            loss = max_llh_loss_covar
            loss_keys = ["ff_mean", "ff_covar"]
        elif do_var:
            loss = max_llh_loss_var
            loss_keys = ["ff_mean", "ff_logvar"]
        else:
            loss = tf.keras.losses.mse
            loss_keys = ["ff_mean"]
    elif ff_loss_str.lower() in ["l1", "mae"]:
        loss = tf.keras.losses.mae
        loss_keys = ["ff_mean"]
    elif ff_loss_str.lower() in ["x-ent", "x_ent"]:
        loss = tf.keras.losses.categorical_crossentropy
        loss_keys = ["ff_mean"]
    else:
        raise NotImplementedError
    return loss, loss_keys


def get_loss_and_keys_histograms(hist_loss_str, smoothing_empl=None):
    """
    Returns the loss function for the SCD histogram estimation.
    :param hist_loss_str: string specifying histogram loss
    :param smoothing_empl: scalar determining the smoothing for Earth Mover's Pinball loss
    :return: loss function, list of keys required (apart from true label, which is always assumed to be first input)
    """
    loss_keys = ["hist"]
    if hist_loss_str.lower() in ["l2", "mse"]:
        def loss(y_true, y_pred): return tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred), 1)  # avg. over channels
    elif hist_loss_str.lower() in ["l1", "mae"]:
        def loss(y_true, y_pred): return tf.reduce_mean(tf.keras.losses.mae(y_true, y_pred), 1)  # avg. over channels
    elif hist_loss_str.lower() in ["x-ent", "x_ent"]:
        def loss(y_true, y_pred): return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred), 1)
    elif hist_loss_str.lower() in ["em1", "em_1"]:
        def loss(y_true, y_pred): return emd_loss(y_true, y_pred, r=1)
    elif hist_loss_str.lower() in ["em2", "em_2"]:
        def loss(y_true, y_pred): return emd_loss(y_true, y_pred, r=2)
    elif hist_loss_str.lower() == "cjs":
        loss = cjs_loss
    elif hist_loss_str.lower() == "empl":
        def loss(y_true, y_pred, tau): return empl(y_true, y_pred, tau, smoothing=smoothing_empl)
        loss_keys += ["tau"]
    else:
        raise NotImplementedError
    return loss, loss_keys


############################
# FLUX FRACTION LOSSES
############################
def max_llh_loss_covar(y_true, y_pred, covar, eps=None):
    """
    (Neg.) maximum likelihood loss function for a full Gaussian covariance matrix.
    :param y_true: label
    :param y_pred: prediction
    :param covar: uncertainty covariance matrix
    :param eps: small number for numerical stability, defaults to tf.keras.backend.epsilon()
    :return: max. likelihood loss (up to a constant)
    """
    if eps is None:
        eps = tf.keras.backend.epsilon()
    err = tf.expand_dims(y_pred - y_true, -1)
    term1 = tf.squeeze(err * tf.linalg.matmul(tf.linalg.inv(covar), err), -1)
    term2 = tf.math.log(eps + tf.linalg.det(covar))
    max_llh_loss = (tf.reduce_sum(term1, 1) + term2) / 2.0
    return max_llh_loss


def max_llh_loss_var(y_true, y_pred, logvar):
    """
    (Neg.) maximum likelihood loss function for a diagonal Gaussian covariance matrix.
    :param y_true: label
    :param y_pred: prediction
    :param logvar: uncertainty log-variances
    :return: max. likelihood loss (up to a constant)
    """
    err = y_pred - y_true
    precision = tf.exp(-logvar)
    term1 = err ** 2 * precision
    term2 = logvar
    max_llh_loss = tf.reduce_sum(term1 + term2, 1) / 2.0
    return max_llh_loss


############################
# HISTOGRAM LOSSES
############################
def emd_loss(y_true, y_pred, r=2, weights=None, do_root=False):
    """
    Computes the Earth Mover's Distance loss.
    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv:1611.05916.
    :param y_true: a 2-D (or 3-D) `Tensor` of the ground truth probability mass functions
    :param y_pred: a 2-D (or 3-D) `Tensor` of the estimated p.m.f.-s
    :param r: a constant for the r-norm.
    :param weights: weight the loss differently for different samples
    :param do_root: if True: raise result to the power of "1/r"
    `y_true` and `y_pred` are assumed to have equal mass as
    \sum^{N}_{i=1} {y_true}_i = \sum^{N}_{i=1} {y_pred}_i
    :return: A 0-D `Tensor` with EMD loss.
    """
    ecdf_true = tf.math.cumsum(y_true, axis=1)
    ecdf_pred = tf.math.cumsum(y_pred, axis=1)
    if weights is None:
        weights = tf.ones_like(ecdf_true)
    if len(weights.shape) < len(y_true.shape):  # if bin-dimension is missing
        weights = tf.expand_dims(weights, 1)
    if r == 1:
        emd = tf.reduce_mean(tf.abs(ecdf_true - ecdf_pred) * weights, axis=1)
    elif r == 2:
        emd = tf.reduce_mean((ecdf_true - ecdf_pred) ** 2 * weights, axis=1)
        if do_root:
            emd = tf.sqrt(emd)
    else:
        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_true - ecdf_pred) * weights, r), axis=1)
        if do_root:
            emd = tf.pow(emd, 1 / r)
    return tf.reduce_mean(emd, 1)  # average over channels


def cjs_loss(y_true, y_pred, eps=1e-10):
    """
    Computes the symmetrical discrete cumulative Jensen-Shannon divergence from https://arxiv.org/pdf/1708.07089.pdf
    :param y_true: labels
    :param y_pred: prediction
    :param eps: lower cutoff for logarithm (for numerical stability)
    :return CJS loss
    """
    cdf_true = tf.cumsum(y_true, axis=1)
    cdf_pred = tf.cumsum(y_pred, axis=1)

    def accjs(p_, q_):
        # if p(i) = 0 then ACCJS(p, q)(i) = 0 since xlog(x) -> 0 as x-> 0
        p_ = tf.clip_by_value(p_, eps, 1.0)
        return 0.5 * tf.reduce_sum(p_ * tf.math.log(p_ / (0.5 * (p_ + q_))), axis=1)

    loss = accjs(cdf_pred, cdf_true) + accjs(cdf_true, cdf_pred)
    return tf.reduce_mean(loss, 1)  # average over channels


def empl(y_true, y_pred, tau, weights=None, smoothing=0.0):
    """
    Compute the Earth Mover's Pinball Loss (arXiv:2106.02051).
    :param y_true: label
    :param y_pred: prediction
    :param tau: quantile levels of interest
    :param weights: weight the loss differently for different samples
    :param smoothing: scalar >= 0 that determines smoothing of loss function around 0
    :return Earth Mover's Pinball Loss

    """
    ecdf_true = tf.math.cumsum(y_true, axis=1)
    ecdf_pred = tf.math.cumsum(y_pred, axis=1)
    delta = ecdf_pred - ecdf_true

    # If there is an extra dimension for the channel: tau might need to be expanded
    if len(tau.shape) == 2 and len(delta.shape) == 3:
        tau = tf.expand_dims(tau, 2)

    # Non-smooth C0 loss (default)
    if smoothing == 0.0:
        mask = tf.cast(tf.greater_equal(delta, tf.zeros_like(delta)), tf.float32) - tau
        loss = mask * delta

    # Smooth loss
    else:
        loss = -tau * delta + smoothing * tf.math.softplus(delta / smoothing)

    if weights is None:
        weights = tf.ones_like(ecdf_true)

    if len(weights.shape) < len(y_true.shape):  # if bin-dimension is missing
        weights = tf.expand_dims(weights, 1)

    # avg. the weighted loss over the bins (1) and channel dimension (2)
    return tf.reduce_mean(loss * weights, [1, 2])
