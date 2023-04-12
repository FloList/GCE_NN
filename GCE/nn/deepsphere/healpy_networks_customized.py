import numpy as np
import tensorflow as tf
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from . import healpy_layers_customized as hp_nn


class HealpyGCNN:
    """
    A graph convolutional network.
    NOTE: when using subclassing of tf.keras.Models, summary() etc. are currently not available (TF <= 2.5), therefore
    we use the Keras functional API here.
    """
    def __init__(self, which, params, index_dict):
        """
        Initializes a graph convolutional neural network using the healpy pixelization scheme
        :param which: submodel, must be one of "flux_fractions" or "histograms"
        :param params: parameter dictionary
        :param index_dict: dictionary containing 'indexes', 'indexes_extended', and 'ind_holes_to_ex' for the ROI at
        each nside hierarchy level
        """
        super(HealpyGCNN, self).__init__()

        self.which = which
        self._p = params
        self._inds = index_dict["indexes"]
        self._inds_ex = index_dict["indexes_extended"]
        self._ind_holes_to_ex = index_dict["ind_holes_to_ex"]

        if which == "flux_fractions":
            dim_out = self._p.mod["n_models"]
            if self._p.nn.ff["alea_covar"]:
                dim_out += dim_out * (dim_out + 1) // 2  # aleatoric uncertainty covariance matrix
            elif self._p.nn.ff["alea_var"]:
                dim_out += dim_out  # aleatoric uncertainty variances
        elif which == "histograms":
            if self._p.nn.hist["continuous"]:
                dim_out = [1, self._p.nn["label_shape"][1][1]]  # only 1 number per PS template: 1 x n_templates
            else:
                dim_out = self._p.nn["label_shape"][1]  # n_bins x n_templates
        else:
            raise NotImplementedError

        self.dim_out = dim_out
        self.dim_out_flat = np.product(dim_out)

    def compute_output(self, input_tensor, tau=None, normed_flux_queries=None):
        """
        Iteratively define the layers of the neural network.
        :param input_tensor: input tensor, can be tf.keras.Inputs
        :param tau: quantile levels for Earth Mover's pinball loss (only concerns SCD histograms)
        :param normed_flux_queries: for continuous dN/dF with EMPL: the fluxes that are queried, scaled to [0, 1]  # TODO: IMPLEMENT!
        :return output dictionary, preprocessed input (can be reused)
        """
        pa = self._p.nn.arch

        need_concat = False  # flag indicating if 2nd(+) channel (which is NOT preprocessed!) needs to be concatenated

        # If t has no channel dimension: add dimension
        if len(input_tensor.shape) == 2:
            first_channel = tf.expand_dims(input_tensor, -1)
        else:
            # If 2nd channel: split up raw input (1st channel) as no preprocessing needs to be done for the rest
            if input_tensor.shape[2] > 1:
                first_channel, other_channels = input_tensor[:, :, :1], input_tensor[:, :, 1:]
                need_concat = True
            else:
                first_channel = input_tensor

        # Get total counts
        tot_counts = tf.reduce_sum(first_channel, axis=1, keepdims=True)

        # Relative counts (i.e., divide by total number of counts in the map)?
        rel_counts = self._p.nn.hist["rel_counts"] if self.which == "histograms" else self._p.nn.ff["rel_counts"]
        if rel_counts:
            first_channel = first_channel / tot_counts

        preprocessed_input = first_channel  # store in a variable that will be returned (input for residual calculation)

        # Concatenate the other channels again
        if need_concat:
            t = tf.concat([preprocessed_input, other_channels], axis=2)
        else:
            t = preprocessed_input

        # Graph-convolutional blocks
        for il in range(len(pa["F"])):

            # Get variables at current resolution
            current_nside = self._p.nn.arch["nsides"][il]
            next_nside = self._p.nn.arch["nsides"][il + 1]
            assert next_nside >= 1, "NN is trying to reduce the resolution below nside=1! Aborting..."
            current_indices = self._inds[il]

            # Chebyshev or monomial convolution?
            conv_layer = hp_nn.HealpyChebyshev if pa["conv"] == "chebyshev5" else hp_nn.HealpyMonomial
            conv_layer_type = "CHEBY" if pa["conv"] == "chebyshev5" else "MONO"

            # ResNet block
            if pa["is_resnet"][il]:
                f_in = 1 if il == 0 else pa["F"][il - 1]
                assert pa["F"][il] == f_in, \
                    "Changing the number of filters with a ResNet block is currently not supported!"
                layer_spec = hp_nn.Healpy_ResidualLayer(layer_type=conv_layer_type,
                                                        layer_kwargs={"K": pa["K"][il], "use_bias": True,
                                                                       "use_bn": pa["batch_norm"][il],
                                                                       "activation": pa["activation"]})
            # Convolutional layer
            else:
                layer_spec = conv_layer(K=pa["K"][il], Fout=pa["F"][il], use_bias=True,
                                        use_bn=pa["batch_norm"][il], activation=pa["activation"])

            # for current_nside < 4: need to reduce n_neighbors and manually set kernel_width:
            if current_nside >= 4:
                n_neighbors = 8  # choose 8 neighbors for Healpix-discretized maps
                kwargs_d = {}
            elif current_nside == 2:
                n_neighbors = 3
                kwargs_d = {"kernel_width": 0.02500 * 16}  # interpolated
            elif current_nside == 1:
                n_neighbors = 2
                kwargs_d = {"kernel_width": 0.02500 * 32}  # interpolated
            else:
                raise NotImplementedError

            # Now, need to append the actual layer
            sphere = SphereHealpix(**kwargs_d, subdivisions=current_nside, indexes=current_indices, nest=True,
                                   k=n_neighbors, lap_type='normalized')
            current_laplacian = sphere.L
            t = layer_spec.get_layer(current_laplacian)(t)

            # Pooling layer
            if pa["pool"] in ["max", "avg"]:
                # Determine the pooling factor
                pool_fac = int(np.log2(current_nside / next_nside))
                t = hp_nn.HealpyPoolWithHoles(p=pool_fac, i=il, inds=self._inds, inds_ex=self._inds_ex,
                                                        ind_holes_to_ex=self._ind_holes_to_ex,
                                                        pool_type=pa["pool"].upper())(t)
            else:
                raise NotImplementedError

        # Flatten (or if convolutions don't go all the way down to a single pixel: average over the pixels
        t = tf.reduce_mean(t, axis=1, keepdims=False)

        # Append total counts?
        if pa["append_tot_counts"] and rel_counts:
            t = tf.concat([t, tf.math.log(tf.squeeze(tot_counts, 1)) / 2.302], axis=1)  # 2.3026 ~ log_e(10)

        # If Earth Mover's pinball loss: append tau at this point
        # We use the scaling proposed by
        # github.com/facebookresearch/SingleModelUncertainty/blob/master/aleatoric/regression/joint_estimation.py
        if tau is not None:
            t = tf.concat([t, (tau - 0.5) * 12], axis=1)

            # Also append flux queries if needed
            if normed_flux_queries is not None:
                t = tf.concat([t, (normed_flux_queries - 0.5) * 12], axis=1)

        # Fully-connected blocks
        # Note: batch norm is provided in a single array for conv. layers and FC layers!
        monotonicity_constraints = 0.0

        for il in range(len(pa["M"])):
            constraint = False
            do_groupsort = False
            this_activation = pa["activation"]

            # if tau is not None and self._p.nn.hist["enforce_monotonicity"]:
            #     # enforce_monotonicity_for_final = 1 + len(self._p.nn.hist["hist_templates"]) \
            #     #     if self._p.nn.hist["continuous"] else 1   # tau and f_queries else only tau
            #     monotonicity_constraints = tf.reduce_sum(t[:, -1 - len(self._p.nn.hist["hist_templates"]):], axis=-1, keepdims=True)
            #     this_activation = None
            #     constraint = True
            #     do_groupsort = True

            t = hp_nn.FullyConnectedBlock(Fout=pa["M"][il], use_bias=True,
                                          use_bn=pa['batch_norm'][len(pa["F"]) + il],
                                          activation=this_activation, constraint=constraint, do_groupsort=do_groupsort)(t)

        # Final fully-connected layer without activation  TODO!!!
        # if self._p.nn.hist["enforce_monotonicity"]:
        #     t = hp_nn.ConstraintDense(self.dim_out_flat, use_bias=False, do_groupsort=False)(t)
        # else:
        t = tf.keras.layers.Dense(self.dim_out_flat, use_bias=False)(t)

        # For histograms: reshape to n_batch x n_bins x n_hist_templates
        if self.which == "histograms":
            t = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], *self.dim_out]))(t)

        # Final layer: will be taken care of when building the model
        kwargs_final = {}
        if self._p.nn.hist["enforce_monotonicity"]:
            kwargs_final["monotonicity_constraints"] = monotonicity_constraints

        out_dict = hp_nn.FinalLayer(which=self.which, params=self._p)(t, **kwargs_final)

        print("The resolution will be successively reduced from nside={:} to nside={:} during a forward pass.".format(
            self._p.data["nside"], self._p.nn.arch["nsides"][-1]), flush=True)

        # Also store tau
        if tau is not None:
            out_dict["tau"] = tau

            if normed_flux_queries is not None:
                out_dict["f_query"] = normed_flux_queries

        return out_dict, preprocessed_input
