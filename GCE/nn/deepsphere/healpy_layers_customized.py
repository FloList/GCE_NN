import numpy as np

from .gnn_layers_customized import *
from ...tf_ops import normalized_softplus, split_mean_var, split_mean_cov
from .lipschitz_utils import GroupSort


class HealpyPoolWithHoles(Layer):
    """
    A customized pooling layer for healpy maps, which supports ROIs with holes. If at least 1 of the 4 subpixels at a
    hierarchy level lies within the ROI, the respective pixel becomes part of the ROI.
    """

    def __init__(self, p, i, inds, inds_ex, ind_holes_to_ex, pool_type="MAX"):
        """
        Initializes the layer.
        :param p: reduction factor >=1 of the nside
        :param i: hierarchy level
        :param inds: list of indices within the ROI at each hierarchy level
        :param inds_ex: list of indices within the extended ROI at each hierarchy level
        :param ind_holes_to_ex: mapping from ROI indices to extended ROI
        :param pool_type: type of pooling, can be "MAX" or "AVG"
        """

        # This is necessary for every Layer
        super(HealpyPoolWithHoles, self).__init__()

        # check p
        if not p >= 1:
            raise IOError("The reduction factors has to be at least 1!")

        # save variables
        self.p = p
        self.filter_size = int(4 ** p)
        self.i = i
        self.inds = inds
        self.inds_ex = inds_ex
        self.ind_holes_to_ex = ind_holes_to_ex
        self.pool_type = pool_type.upper()

    def filter(self, x):
        """
        Pooling operation
        :param x: input
        :return pooled tensor
        """

        if self.p > 0:
            smallest_n_contiguous = len(self.inds_ex[self.i])
            x_transp = tf.transpose(x, [1, 0, 2])  # M, n_batch, F

            # for max pooling: pixels outside ROI are -infinity -> don't affect maximum
            if self.pool_type.upper() == "MAX":
                dummy = -np.infty * tf.ones((smallest_n_contiguous, tf.shape(x)[0], x.shape[2]))  # M_ext, n_batch, F

            # for avg. pooling: pixels outside ROI are nan -> use nanmean
            elif self.pool_type.upper() == "AVG":
                dummy = np.nan * tf.ones((smallest_n_contiguous, tf.shape(x)[0], x.shape[2]))  # M_ext, n_batch, F

            else:
                raise NotImplementedError

            # Fill x_transp into dummy at the right indices
            x_scattered = tf.tensor_scatter_nd_update(dummy, tf.expand_dims(self.ind_holes_to_ex[self.i], -1), x_transp)
            x_scattered_transp = tf.transpose(x_scattered, [1, 0, 2])  # n_batch, M_ext, F

            # Make blocks: tensor size is n_batch, filter size, M // filter size, F
            x_scattered_blocks = tf.transpose(tf.reshape(x_scattered_transp,
                                                         [tf.shape(x)[0], -1, self.filter_size, x.shape[2]]),
                                              [0, 2, 1, 3])

            # Pool each block
            if self.pool_type == "MAX":
                x_pooled = tf.reduce_max(x_scattered_blocks, axis=1, keepdims=False)
            elif self.pool_type == "AVG":
                x_pooled = tf.experimental.numpy.nanmean(x_scattered_blocks, axis=1, keepdims=False)
            else:
                raise NotImplementedError

            # Get pixels in ROI at the next coarser hierarchy level
            x_out = tf.gather_nd(x_pooled, tf.tile(tf.expand_dims(tf.expand_dims(
                self.ind_holes_to_ex[self.i + 1], 0), -1), [tf.shape(x)[0], 1, 1]), batch_dims=1)

            return x_out

        else:
            return x

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """
        pass

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """
        return self.filter(input_tensor)


class HealpyPool(Layer):
    """
    A pooling layer for healpy maps, makes use of the fact that a pixels is always divided into 4 subpixels when
    increasing the nside of a HealPix map
    """

    def __init__(self, p, pool_type="MAX", **kwargs):
        """
        initializes the layer
        :param p: reduction factor >=1 of the nside -> number of nodes reduces by 4^p, note that the layer only checks
                  if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct
                  (should be nested ordering)
        :param pool_type: type of pooling, can be "MAX" or "AVG"
        :param kwargs: additional kwargs passed to the keras pooling layer
        """
        # This is necessary for every Layer
        super(HealpyPool, self).__init__()

        # check p
        if not p >= 1:
            raise IOError("The reduction factors has to be at least 2!")

        # save variables
        self.p = p
        self.filter_size = int(4**p)
        self.pool_type = pool_type
        self.kwargs = kwargs

        if pool_type == "MAX":
            self.filter = tf.keras.layers.MaxPool1D(pool_size=self.filter_size, strides=self.filter_size,
                                                    padding='valid', data_format='channels_last', **kwargs)
        elif pool_type == "AVG":
            self.filter = tf.keras.layers.AveragePooling1D(pool_size=self.filter_size, strides=self.filter_size,
                                                            padding='valid', data_format='channels_last', **kwargs)
        else:
            raise IOError(f"Pooling type not understood: {self.pool_type}")

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        n_nodes = int(input_shape[1])
        if n_nodes % self.filter_size != 0:
            raise IOError("Input shape {input_shape} not compatible with the filter size {self.filter_size}")

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        return self.filter(input_tensor)


class HealpyPseudoConv(Layer):
    """
    A pseudo convolutional layer on Healpy maps. It makes use of the Healpy pixel scheme and reduces the nside by
    averaging the pixels into bigger pixels using learnable weights
    """

    def __init__(self, p, Fout, kernel_initializer=None, **kwargs):
        """
        initializes the layer
        :param p: reduction factor >=1 of the nside -> number of nodes reduces by 4^p, note that the layer only checks
                  if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct
                  (should be nested ordering)
        :param Fout: number of output channels
        :param kernel_initializer: initializer for kernel init
        :param kwargs: additional keyword arguments passed to the keras 1D conv layer
        """
        # This is necessary for every Layer
        super(HealpyPseudoConv, self).__init__()

        # check p
        if not p >= 1:
            raise IOError("The reduction factors has to be at least 1!")

        # save variables
        self.p = p
        self.filter_size = int(4 ** p)
        self.Fout = Fout
        self.kernel_initializer = kernel_initializer
        self.kwargs = kwargs

        # create the files
        self.filter = tf.keras.layers.Conv1D(self.Fout, self.filter_size, strides=self.filter_size,
                                             padding='valid', data_format='channels_last',
                                             kernel_initializer=self.kernel_initializer, **self.kwargs)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        n_nodes = int(input_shape[1])
        if n_nodes % self.filter_size != 0:
            raise IOError(f"Input shape {input_shape} not compatible with the filter size {self.filter_size}")
        self.filter.build(input_shape)

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        return self.filter(input_tensor)


class HealpyPseudoConv_Transpose(Layer):
    """
    A pseudo transpose convolutional layer on Healpy maps. It makes use of the Healpy pixel scheme and increases
    the nside by applying a transpose convolution to the pixels into bigger pixels using learnable weights
    """

    def __init__(self, p, Fout, kernel_initializer=None, **kwargs):
        """
        initializes the layer
        :param p: Boost factor >=1 of the nside -> number of nodes increases by 4^p, note that the layer only checks
                  if the dimensionality of the input is evenly divisible by 4^p and not if the ordering is correct
                  (should be nested ordering)
        :param Fout: number of output channels
        :param kernel_initializer: initializer for kernel init
        :param kwargs: additional keyword arguments passed to the keras transpose conv layer
        """
        # This is necessary for every Layer
        super(HealpyPseudoConv_Transpose, self).__init__()

        # check p
        if not p >= 1:
            raise IOError("The boost factors has to be at least 1!")

        # save variables
        self.p = p
        self.filter_size = int(4 ** p)
        self.Fout = Fout
        self.kernel_initializer = kernel_initializer
        self.kwargs = kwargs

        # create the files
        self.filter = tf.keras.layers.Conv2DTranspose(self.Fout, (1, self.filter_size), strides=(1, self.filter_size),
                                                      padding='valid', data_format='channels_last',
                                                      kernel_initializer=self.kernel_initializer, **self.kwargs)

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """

        input_shape = list(input_shape)
        n_nodes = input_shape[1]
        if n_nodes % self.filter_size != 0:
            raise IOError(f"Input shape {input_shape} not compatible with the filter size {self.filter_size}")

        # add the additional dim
        input_shape.insert(1, 1)

        self.filter.build(input_shape)

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        input_tensor = tf.expand_dims(input_tensor, axis=1)
        return tf.squeeze(self.filter(input_tensor), axis=1)


class HealpyChebyshev():
    """
    A helper class for a Chebyshev5 layer using healpy indices instead of the general Layer
    """
    def __init__(self, K, Fout=None, initializer=None, activation=None, use_bias=False,
                 use_bn=0, **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm (1) or instance norm (2) before adding the bias (0 otherwise)
        :param kwargs: additional keyword arguments passed on to add_weight
        """
        # we only save the variables here
        self.K = K
        self.Fout = Fout
        self.initializer = initializer
        self.activation = activation
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.kwargs = kwargs

    def get_layer(self, L):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :return: Chebyshev5 layer that can be called
        """

        # now we init the layer
        return Chebyshev(L=L, K=self.K, Fout=self.Fout, initializer=self.initializer, activation=self.activation,
                          use_bias=self.use_bias, use_bn=self.use_bn, **self.kwargs)


class HealpyMonomial():
    """
    A graph convolutional layer using Monomials
    """
    def __init__(self, K, Fout=None, initializer=None, activation=None, use_bias=False, use_bn=0, **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm (1) or instance norm (2) before adding the bias (0 otherwise)
        :param kwargs: additional keyword arguments passed on to add_weight
        """

        # we only save the variables here
        self.K = K
        self.Fout = Fout
        self.initializer = initializer
        self.activation = activation
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.kwargs = kwargs

    def get_layer(self, L):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :return: Monomial layer that can be called
        """

        # now we init the layer
        return Monomial(L=L, K=self.K, Fout=self.Fout, initializer=self.initializer, activation=self.activation,
                        use_bias=self.use_bias, use_bn=self.use_bn, **self.kwargs)


class Healpy_ResidualLayer():
    """
    A generic residual layer of the form
    in -> layer -> layer -> out + in
    with optional batchnorm in the end
    """

    def __init__(self, layer_type, layer_kwargs, activation=None, act_before=False, use_bn=False, bn_kwargs=None,
                 alpha=1.0):
        """
        Initializes the residual layer with the given argument
        :param layer_type: The layer type, either "CHEBY" or "MONO" for chebychev or monomials
        :param layer_kwargs: A dictionary with the inputs for the layer
        :param activation: activation function to use for the res layer
        :param act_before: use activation before skip connection
        :param use_bn: use batch norm in between the layers
        :param bn_kwargs: An optional dictionary containing further keyword arguments for the normalization layer
        :param alpha: Coupling strength of the input -> layer(input) + alpha*input
        """

        # we only save the variables here
        self.layer_type = layer_type
        self.layer_kwargs = layer_kwargs
        self.activation = activation
        self.act_before = act_before
        self.use_bn = use_bn
        self.bn_kwargs = bn_kwargs
        self.alpha = alpha

    def get_layer(self, L):
        """
        initializes the actual layer, should be called once the graph Laplacian has been calculated
        :param L: the graph laplacian
        :return: GCNN_ResidualLayer layer that can be called
        """
        # we add the graph laplacian to all kwargs
        self.layer_kwargs.update({"L": L})

        return GCNN_ResidualLayer(layer_type=self.layer_type, layer_kwargs=self.layer_kwargs,
                                  activation=self.activation, act_before=self.act_before,
                                  use_bn=self.use_bn, bn_kwargs=self.bn_kwargs, alpha=self.alpha)


class FullyConnectedBlock(Layer):
    """
    A fully-connected block.
    """
    def __init__(self, Fout, initializer=None, activation=None, use_bias=False, use_bn=0, constraint=False, do_groupsort=False, **kwargs):
        """
        Initializes the fully-connected block
        :param Fout: Number of features (channels) of the output
        :param initializer: initializer to use for weight initialisation
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm (1) or instance norm (2) before adding the bias (0 otherwise)
        :param constraint: constrain the MaxNorm of the kernel?
        :param kwargs: additional keyword arguments passed on to tf.keras.Layer
        """

        # This is necessary for every Layer
        super(FullyConnectedBlock, self).__init__(**kwargs)

        # save necessary params
        self.Fout = Fout
        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn == 1:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, center=False, scale=False)
        elif self.use_bn == 2:
            self.inst_norm = tf.keras.layers.Lambda(instance_normalization)
        self.initializer = initializer
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
        self.kwargs = kwargs
        super(FullyConnectedBlock, self).__init__()

        self.Fout = Fout
        self.kwargs = kwargs
        self.constraint = constraint
        self.do_groupsort = do_groupsort

    def build(self, input_shape):
        """
        Build the weights of the layer
        :param input_shape: shape of the input, batch dim has to be defined
        """
        kernel_initializer = "glorot_uniform" if self.initializer is None else self.initializer
        if self.constraint:
            # self.fc = tf.keras.layers.Dense(self.Fout, use_bias=self.use_bias, kernel_initializer=kernel_initializer)  # TODO!!! MONOTONIC!
            self.fc = ConstraintDense(self.Fout, kernel_initializer=kernel_initializer, do_groupsort=self.do_groupsort)

        else:
            self.fc = tf.keras.layers.Dense(self.Fout, use_bias=self.use_bias, kernel_initializer=kernel_initializer)

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """
        x = self.fc(input_tensor)
        if self.use_bn == 1:
            x = self.bn(x)
        elif self.use_bn == 2:
            x = self.inst_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FinalLayer(Layer):
    """
    Final neural network layer for flux fraction / SCD histogram submodel.
    """
    def __init__(self, which, params, **kwargs):
        """
        Initializes the final layer.
        :param which: submodel, must be one of "flux_fractions" or "histograms"
        :param params: parameter dictionary
        :param kwargs: additional keyword arguments
        """

        # This is necessary for every Layer
        super(FinalLayer, self).__init__(**kwargs)

        # save necessary params
        self._which = which
        self._p = params

        if self._which == "flux_fractions":
            last_act = self._p.nn.ff["last_act"]
        elif self._which == "histograms":
            last_act = self._p.nn.hist["last_act"]
        else:
            raise NotImplementedError

        if last_act == "softmax":
            self.activation = lambda x: tf.nn.softmax(x, axis=1)
        elif last_act == "normalized_softplus":
            self.activation = lambda x: normalized_softplus(x, axis=1)
        elif last_act == "sigmoid":  # note: this is for the continuous case!
            self.activation = lambda x: tf.nn.sigmoid(x)
        else:
            raise NotImplementedError

    def call(self, input_tensor, *args, **kwargs):
        """
        Calls the layer on an input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """
        x = input_tensor
        output_dict = {}

        # Flux fraction submodel:
        if self._which == "flux_fractions":

            # Aleatoric uncertainty covariance matrix
            if self._p.nn.ff["alea_covar"]:
                output_dict["ff_mean"], output_dict["ff_covar"] = split_mean_cov(x, self.activation,
                                                                                 self._p.mod["n_models"], eps=1e-6)

            # Aleatoric uncertainty variances
            elif self._p.nn.ff["alea_var"]:
                output_dict["ff_mean"], output_dict["ff_logvar"] = split_mean_var(x, self.activation,
                                                                                  self._p.mod["n_models"])

            # Only mean estimates
            else:
                x = self.activation(x)
                output_dict["ff_mean"] = x

        # Histogram submodel:
        else:
            if self._p.nn.hist["enforce_monotonicity"]:
                assert "monotonicity_constraints" in kwargs.keys()
                x += 0.0 * kwargs["monotonicity_constraints"]  # TODO!!

            x = self.activation(x)
            output_dict["hist"] = x
        return output_dict


class PoissonResidualLayer(Layer):
    def __init__(self, params, temp_dict, **kwargs):
        """
        This layer computes the residual after removing the best-fit estimate of the Poissonian templates.
        :param preprocessed_input_maps: input photon-count maps (already divided by total counts if using rel. counts)
        :param params: parameter dictionary
        :param temp_dict: template dictionary
        :param kwargs: additional keyword arguments
        """
        super(PoissonResidualLayer, self).__init__(**kwargs)

        self.remove_exp = params.nn["remove_exp"]

        # Convert T_counts and counts-to-flux dict. to an array in the order of the models
        self.poiss_inds = np.argwhere([temp in params.mod["models_P"] for temp in params.mod["models"]]).flatten()
        self.t_counts_compressed_arr = np.asarray([temp_dict["T_counts"][temp] for temp in params.mod["models"]]
                                                  ).astype(np.float32)[:, temp_dict["indices_roi"]]
        self.counts2flux_roi_arr = np.asarray([temp_dict["counts_to_flux_ratio_roi"][temp]
                                               for temp in params.mod["models"]]).astype(np.float32)

        self.rescale_compressed_f32_exp = tf.expand_dims(temp_dict["rescale_compressed"].astype(np.float32), 0)

    def call(self, inputs, *args, **kwargs):
        """
        Calls the layer on input tensors
        :param inputs: consists of
            1. preprocessed_input_maps: input photon-count maps (already divided by total counts if using rel. counts)
            2. ff_mean: prediction for the flux fractions
        :param args: further arguments
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """
        preprocessed_input_maps, ff_mean = inputs

        # Only 1 channel: squeeze
        preprocessed_input_maps = tf.squeeze(preprocessed_input_maps, 2)

        # From flux fractions, get count fractions
        ff_sum = tf.reduce_sum(ff_mean, 1, keepdims=True)
        count_fracs_unnorm = ff_mean * tf.expand_dims(self.counts2flux_roi_arr, 0)
        count_fracs = count_fracs_unnorm / (tf.reduce_sum(count_fracs_unnorm, axis=1, keepdims=True) / ff_sum)

        # Get counts per template
        if self.remove_exp:  # if exposure is removed by dividing by exp / mean(exp): convert to count maps
            count_maps = preprocessed_input_maps * self.rescale_compressed_f32_exp
        else:  # if count maps are shown to the NN, leave as it is
            count_maps = preprocessed_input_maps

        total_counts = tf.reduce_sum(count_maps, 1, keepdims=True)
        counts_per_temp = total_counts * count_fracs

        # Get ratio between counts per template and template sum
        t_counts_rescale_fac = counts_per_temp / tf.reduce_sum(self.t_counts_compressed_arr.T.astype(np.float32),
                                                               0, keepdims=True)

        # Best-fit count maps per template: n_batch x n_models x n_pix
        count_maps_modelled_per_temp = tf.einsum('ij,jk->ijk', t_counts_rescale_fac, self.t_counts_compressed_arr)

        # Sum over the Poissonian models
        count_maps_modelled_poiss = tf.reduce_sum(tf.gather(count_maps_modelled_per_temp, indices=self.poiss_inds,
                                                            axis=1), 1)

        # Calculate the residual
        count_maps_residual_raw = count_maps - count_maps_modelled_poiss  # 1 channel

        # If needed: remove exposure again
        if self.remove_exp:
            count_maps_residual = count_maps_residual_raw / self.rescale_compressed_f32_exp
        else:
            count_maps_residual = count_maps_residual_raw

        return count_maps_residual


# Variant 1: with positive weights
# class MonotonicDense(tf.keras.layers.Layer):
#     def __init__(self, Fout, kernel_initializer=None, monotonic_inputs=0, use_bias=True, activation=None, use_bn=0,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.Fout = Fout
#         self.monotonic_inputs = monotonic_inputs
#         self.initializer = "glorot_uniform" if kernel_initializer is None else kernel_initializer
#         self.use_bias = use_bias
#         self.use_bn = use_bn
#
#         if activation is None or callable(activation):
#             self.activation = activation
#         elif hasattr(tf.keras.activations, activation):
#             self.activation = getattr(tf.keras.activations, activation)
#         else:
#             raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")
#
#     def build(self, input_shape):
#         super().build(input_shape)
#         self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.Fout),
#                                       trainable=True,
#                                       constraint=lambda w: tf.where(tf.tile(tf.range(input_shape[1])[:, None],
#                                                                             (1, w.shape[1]))
#                                                                     > input_shape[1] - self.monotonic_inputs,
#                                                                     tf.keras.constraints.NonNeg()(w),
#                                                                     tf.keras.constraints.Constraint()(w)))
#         # initially, need to manually apply a ReLU because initializer has priority over constraints
#         w_raw = self.get_weights()[0]
#         w_raw[-self.monotonic_inputs:, :] = np.clip(w_raw[-self.monotonic_inputs:, :], 0, np.infty)
#         self.set_weights([w_raw])
#
#         if self.use_bias:
#             self.bias = self.add_weight(name='bias', shape=(self.Fout,), initializer='zeros', trainable=True)
#
#     def call(self, inputs):
#         output = tf.matmul(inputs, self.kernel)
#
#         if self.use_bias:
#             output = tf.nn.bias_add(output, self.bias)
#
#         if self.use_bn == 1:
#             output = self.bn(output)
#         elif self.use_bn == 2:
#             output = self.inst_norm(output)
#
#         if self.activation is not None:
#             output = self.activation(output)
#
#         return output

# Variant 2: with Lipschitz constraint
class ConstraintDense(tf.keras.layers.Layer):
    def __init__(self, Fout, kernel_initializer=None, use_bias=True, do_groupsort=False, **kwargs):
        super().__init__(**kwargs)
        self.Fout = Fout
        self.initializer = "glorot_uniform" if kernel_initializer is None else kernel_initializer
        self.use_bias = use_bias
        self.do_groupsort = do_groupsort

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.Fout),
                                      trainable=True,
                                      constraint=tf.keras.constraints.MaxNorm(max_value=np.infty))  # TODO!!!
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(self.Fout,), initializer='zeros', trainable=True)

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.do_groupsort:
            output = GroupSort(n_groups=2)(output)  # TODO

        return output