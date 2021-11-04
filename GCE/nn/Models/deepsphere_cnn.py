import numpy as np
import tensorflow as tf
import os
from ..deepsphere.healpy_networks_customized import HealpyGCNN
from ..deepsphere.healpy_layers_customized import PoissonResidualLayer


class DeepsphereCNN:
    def __init__(self, params, index_dict, temp_dict, strategy):
        """
        Initialise the DeepSphere-based CNN.
        :param params: parameter dictionary
        :param index_dict: dictionary containing 'indexes', 'indexes_extended', and 'ind_holes_to_ex' for the ROI at
        each nside hierarchy level
        :param temp_dict: dictionary containing the templates, needed for the residual computation
        :strategy: TF parallelization strategy
        """
        self._p = params
        self._index_dict = index_dict
        self._template_dict = temp_dict
        self._out = dict()
        self._strategy = strategy

    def build_model(self):
        """
        Defines and builds the neural network.
        :return neural network, trainable weights dictionary
        """
        with self._strategy.scope():
            # Define the input tensor
            input_tensor = tf.keras.Input(shape=(self._p.nn["input_shape"]))

            tau = None
            outdict = {}

            # Flux fraction submodel
            if self._p.nn.ff["return_ff"]:
                model_ff = HealpyGCNN(which="flux_fractions", params=self._p, index_dict=self._index_dict)
                model_ff_outdict, preprocessed_input = model_ff.compute_output(input_tensor=input_tensor)
                outdict = {**outdict, **model_ff_outdict}

            # SCD histogram submodel
            if self._p.nn.hist["return_hist"]:
                hist_nn_input = input_tensor
                # Feed residual as a second channel?
                if self._p.nn.ff["return_ff"] and self._p.nn.hist["calculate_residual"]:
                    # Get input maps after removing best-fit Poissonian emission
                    poissonian_residual = PoissonResidualLayer(self._p, self._template_dict)([
                        preprocessed_input, model_ff_outdict["ff_mean"]])
                    # in this case: feed input_tensor with Poissonian residual
                    hist_nn_input = tf.concat([tf.expand_dims(input_tensor, 2),
                                               tf.expand_dims(poissonian_residual, 2)], axis=2)  # 2nd channel: residual

                # Define tau input tensor here:
                if self._p.train["hist_loss"].upper() == "EMPL":
                    tau = tf.keras.Input(shape=1)

                model_hist = HealpyGCNN(which="histograms", params=self._p, index_dict=self._index_dict)
                model_hist_outdict, _ = model_hist.compute_output(input_tensor=hist_nn_input, tau=tau)
                outdict = {**outdict, **model_hist_outdict}

            assert len(outdict) > 0, "Nothing to predict: neither flux fractions nor SCD histograms have been selected!"

            # Now, build the keras model
            if tau is not None:
                model = tf.keras.Model(inputs=[input_tensor, tau], outputs=outdict)
            else:
                model = tf.keras.Model(inputs=input_tensor, outputs=outdict)

            # Print summary
            model.summary()

            # Now: store trainable parameters for each submodel to enable flexible training
            trainable_weights_dict = {"ff": [], "hist": []}
            ff_hist_ind = 0 if self._p.nn.ff["return_ff"] else 1

            # Iterate over the layers and store trainable weights
            for layer in model.layers:
                if len(layer.trainable_weights) > 0:
                    if ff_hist_ind == 0:
                        trainable_weights_dict["ff"].extend(layer.trainable_weights)
                    elif ff_hist_ind == 1:
                        trainable_weights_dict["hist"].extend(layer.trainable_weights)
                    else:
                        raise ValueError

                # Done with FF submodel?
                if 'final_layer' in layer.name:
                    ff_hist_ind += 1

            tot_weights_saved = len(trainable_weights_dict["ff"]) + len(trainable_weights_dict["hist"])
            assert tot_weights_saved == len(model.trainable_weights), \
                "Expected to save {:} weights, but trainable_weights_dict contains {:} weights. Aborting...".format(
                    tot_weights_saved, len(model.trainable_weights))

            # Also store the final dense layers separately, enabling it to be trained while the other weights are frozen
            dense_layers = [layer for layer in model.layers if "dense" in layer.name]
            if len(dense_layers) == 2:
                trainable_weights_dict["ff_final_dense"] = dense_layers[0].trainable_weights
                trainable_weights_dict["hist_final_dense"] = dense_layers[1].trainable_weights
            elif len(dense_layers) == 1:
                if self._p.nn.ff["return_ff"]:
                    trainable_weights_dict["ff_final_dense"] = dense_layers[0].trainable_weights
                else:
                    trainable_weights_dict["hist_final_dense"] = dense_layers[0].trainable_weights
            else:
                raise ValueError("Expected 1 or 2 dense layers outside a fully-connected block, but found {:}!"
                                 "".format(len(dense_layers)))
        return model, trainable_weights_dict