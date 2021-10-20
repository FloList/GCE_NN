"""
This module implements the graph convolutional neural network.
Most of the code is based on https://github.com/mdeff/cnn_graph/.
"""
from __future__ import division

import os
import sys
import time
from builtins import range
from zipfile import ZipFile
from tqdm import tqdm
from gce_concrete_dropout import ConcreteDropout
import numpy as np
import healpy as hp
from scipy import sparse
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import debug as tf_debug
from gce_utils import stack_var_fracs, flatten_var_fracs, combine_list_of_dicts, get_template, mkdir_p, plot_posterior
from EM_distance_tf import emd_loss, cjs_loss
from CDF_quantile_regression_tf import cdf_quantile_loss
from . import utils
tfd = tfp.distributions


# Python 2 compatibility.
if hasattr(time, 'process_time'):
    process_time = time.process_time
else:
    import warnings
    warnings.warn('The CPU time is not working with Python 2.')
    def process_time():
        return np.nan

# def show_all_variables():
#     import tensorflow as tf
#     import tensorflow.contrib.slim as slim
#     model_vars = tf.trainable_variables()
#     slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# # # # # # # # # # # # BASE MODEL # # # # # # # # # # # #
class base_model(object):
    """Common methods for all models."""

    def __init__(self, loss='cross_entropy', **kwargs):
        self.regularizers = []
        self.regularizers_size = []
        self._loss_type = loss
        self.input_train = []
        self.input_test = []
        self.global_step = []

        self.models = None if "models" not in kwargs.keys() else kwargs["models"]
        self.pre_gen = False if "PRE_GEN" not in kwargs.keys() else kwargs["PRE_GEN"]

        self.aleatoric = False if "aleatoric" not in kwargs.keys() else kwargs["aleatoric"]
        self.alea_split = False if "alea_split" not in kwargs.keys() else kwargs["alea_split"]
        self.alea_only_var = False if "alea_only_var" not in kwargs.keys() else kwargs["alea_only_var"]
        self.covar_scaling = 1.0 if "covar_scaling" not in kwargs.keys() else kwargs["covar_scaling"]
        self.epistemic = False if "epistemic" not in kwargs.keys() else kwargs["epistemic"]
        self.epi_l = None if "epi_prior_length_scale" not in kwargs.keys() else kwargs["epi_prior_length_scale"]
        self.epi_n = None if "epi_n_training_samples" not in kwargs.keys() else kwargs["epi_n_training_samples"]
        self.epi_p_init_min = None if "epi_p_init_min" not in kwargs.keys() else kwargs["epi_p_init_min"]
        self.epi_p_init_max = None if "epi_p_init_max" not in kwargs.keys() else kwargs["epi_p_init_max"]
        self.epi_n_dropout = None if "epi_n_dropout" not in kwargs.keys() else kwargs["epi_n_dropout"]
        self.epi_dropout_last_layer = True if "epi_dropout_output_layer" not in kwargs.keys() else kwargs["epi_dropout_output_layer"]
        self.estimate_var_fracs = False if "estimate_var_fracs" not in kwargs.keys() else kwargs["estimate_var_fracs"]
        self.deactivate_hists = False if "deactivate_hists" not in kwargs.keys() else kwargs["deactivate_hists"]
        # Gaussian mixture
        self.gaussian_mixture = False if "gaussian_mixture" not in kwargs.keys() else kwargs["gaussian_mixture"]
        self.n_gaussians = None if "n_gaussians" not in kwargs.keys() else kwargs["n_gaussians"]
        self.truncate_gaussians = False if "truncate_gaussians" not in kwargs.keys() else kwargs["truncate_gaussians"]
        self.distribution_strategy = "diff" if "distribution_strategy" not in kwargs.keys() else kwargs["distribution_strategy"]
        # GCE histograms
        self.gce_return_hist = False if "gce_return_hist" not in kwargs.keys() else kwargs["gce_return_hist"]
        self.gce_only_hist = False if "gce_only_hist" not in kwargs.keys() else kwargs["gce_only_hist"]
        self.gce_hist_n_bins = None if "gce_hist_bins" not in kwargs.keys() else len(kwargs["gce_hist_bins"]) - 1
        self.gce_hist_loss = None if "gce_hist_loss" not in kwargs.keys() else kwargs["gce_hist_loss"]
        self.gce_hist_act = "softmax" if "gce_hist_act" not in kwargs.keys() else kwargs["gce_hist_act"]
        self.gce_hist_tau_dist = "uniform" if "gce_hist_tau_dist" not in kwargs.keys() else kwargs["gce_hist_tau_dist"]
        self.gce_hist_pinball_smoothing = 0.0 if "gce_hist_pinball_smoothing" not in kwargs.keys() else kwargs["gce_hist_pinball_smoothing"]
        self.gce_hist_lambda = None if "gce_hist_lambda" not in kwargs.keys() else kwargs["gce_hist_lambda"]
        self.gce_hist_FF_weights_loss = False if "gce_hist_FF_weights_loss" not in kwargs.keys() else kwargs["gce_hist_FF_weights_loss"]
        self.gce_hist_rel_counts = False if "gce_hist_rel_counts" not in kwargs.keys() else kwargs["gce_hist_rel_counts"]
        self.indexes_extended = None if "indexes_extended" not in kwargs.keys() else kwargs["indexes_extended"]
        self.gce_hist_split = False if "gce_hist_split" not in kwargs.keys() else kwargs["gce_hist_split"]
        self.gce_hist_step = 1 if "gce_hist_step" not in kwargs.keys() else kwargs["gce_hist_step"]

        # Does NN estimate FFs or CFs?
        # U-Net: pixel-wise predictions of count fraction
        if "NN_type" in kwargs.keys() and kwargs["NN_type"] == "U-Net":
            self.NN_estimates_CFs = True
        else:
            if self.pre_gen:  # if data is pre-generated: always: flux fractions are stored with the maps!
                self.NN_estimates_CFs = False
            elif not self.pre_gen:  # if data is generated on-the-fly: depends on "remove_exp"
                self.NN_estimates_CFs = kwargs["remove_exp"]

        # Are the maps divided by exp / mean(exp) before showing them to the NN
        self.remove_exp = False if "remove_exp" not in kwargs.keys() else kwargs["remove_exp"]

        self.gce_hist_templates = None if "gce_hist_templates" not in kwargs.keys() else kwargs["gce_hist_templates"]
        if self.gce_hist_templates is not None:
            self.gce_hist_indices = np.asarray([np.argwhere(hist_temp == np.asarray(self.models)) for hist_temp in self.gce_hist_templates]).flatten()

        # Calculate mapping indexes (with holes) -> indexes extended
        if hasattr(self, "indexes") and self.indexes_extended is not None:
            self.ind_holes_to_ext = [np.asarray([np.argwhere(self.indexes_extended[i] == ind)[0][0] for ind in self.indexes[i]]) \
                        for i in range(len(self.indexes))]

        # Set weight and dropout regularisers
        if self.epistemic and self.epi_l is not None and self.epi_n is not None:
            self.epi_wd = self.epi_l ** 2. / self.epi_n  # weight regulariser
            self.epi_dd = 2. / self.epi_n  # dropout regulariser

        if "pre_fit_fcn" in kwargs.keys():
            self._pre_fit_fcn = kwargs["pre_fit_fcn"]

    # TODO: IMPLEMENT
    def saliency_map(self, name="saliency_map"):
        """
        Produce a saliency map as described in the paper:
        `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
        <https://arxiv.org/abs/1312.6034>`_.
        The saliency map is the gradient of the max element in output w.r.t input.
        Returns:
            tf.Tensor: the saliency map. Has the same shape as input.
        """
        output = tf.split(self.op_prediction, self.op_prediction.shape[1], axis=1)
        input = self.ph_data
        saliency_op = [tf.gradients(out, input) for out in output]  # [:]
        return tf.identity(saliency_op, name=name)

    # High-level interface which runs the constructed computational graph.
    def predict(self, in_dict, sess=None, only_last_layer=False, tau_hist=None):
        data = in_dict["data"]
        labels = None if "label" not in in_dict.keys() else in_dict["label"]
        var_fracs = None if "var_fracs" not in in_dict.keys() else in_dict["var_fracs"]
        gce_hist = None if "gce_hist" not in in_dict.keys() else in_dict["gce_hist"]

        output_dict = dict()
        loss = 0
        size = data.shape[0]
        sess = self.get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, *data.shape[1:]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            if len(data.shape) == 2:
                batch_data[end-begin:] = np.tile(tmp_data[:1, :], [self.batch_size-(end-begin), 1])
            elif len(data.shape) == 3:
                batch_data[end - begin:] = np.tile(tmp_data[:1, :], [self.batch_size - (end - begin), 1, 1])
            feed_dict = {self.ph_data: batch_data, self.ph_training: False, self.only_last_layer: only_last_layer}
            if tau_hist is not None:
                batch_tau_hist = np.ones((self.batch_size, *tau_hist.shape[1:]))
                batch_tau_hist[:end-begin] = tau_hist[begin:end]
                feed_dict[self.ph_tau_hist] = batch_tau_hist
            if var_fracs is not None:
                batch_var_fracs = np.ones((self.batch_size, *var_fracs.shape[1:]))
                batch_var_fracs[:end-begin] = var_fracs[begin:end]
                feed_dict[self.ph_var_fracs] = batch_var_fracs
            if gce_hist is not None:
                batch_gce_hist = np.ones((self.batch_size, *gce_hist.shape[1:]))
                batch_gce_hist[:end-begin] = gce_hist[begin:end]
                feed_dict[self.ph_gce_hist] = batch_gce_hist

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = 1.0 / labels.shape[1] * np.ones((self.batch_size, *labels.shape[1:]))  # don't use zeros here!
                batch_labels[:end-begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred_dict, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred_dict = sess.run(self.op_prediction, feed_dict)

            # For cross-entropy losses: convert logits -> FFs
            if self._loss_type == "x-ent" and "logits_mean" in batch_pred_dict.keys():
                batch_pred_dict["logits_mean"] = np.exp(batch_pred_dict["logits_mean"]) / np.exp(batch_pred_dict["logits_mean"]).sum(1, keepdims=True)
            if self.gce_hist_loss == "x-ent" and "gce_hist" in batch_pred_dict.keys():
                batch_pred_dict["gce_hist"] = np.exp(batch_pred_dict["gce_hist"]) / np.exp(batch_pred_dict["gce_hist"]).sum(1, keepdims=True)

            # Rescale covariance
            if "logvar" in batch_pred_dict.keys():
                if self._loss_type == "l2":
                    batch_pred_dict["covar"] = np.exp(np.asarray([np.diag(batch_pred_dict["logvar"][i] / self.covar_scaling) for i in range(batch_pred_dict["logvar"].shape[0])])) * np.expand_dims(np.eye(batch_pred_dict["logvar"].shape[1]), 0)
                elif self._loss_type == "l1":
                    two_b = np.exp(np.asarray([np.diag(batch_pred_dict["logvar"][i]) for i in range(batch_pred_dict["logvar"].shape[0])]))
                    batch_pred_dict["covar"] = two_b ** 2 / 2.0 * np.expand_dims(np.eye(batch_pred_dict["logvar"].shape[1]), 0)  # var: 2b^2 = (2b)^2 / 2
            if "logits_covar" in batch_pred_dict.keys():
                batch_pred_dict["covar"] /= self.covar_scaling

            for key in batch_pred_dict.keys():
                if key == "logvar":  # only save covariance matrix
                    continue
                if key not in output_dict.keys():
                    output_dict[key] = []
                output_dict[key].append(batch_pred_dict[key][:end-begin])

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key])

        if labels is not None:
            output_dict["loss"] = loss * self.batch_size / size

        return output_dict

    def evaluate(self, out_dict, sess=None, only_last_layer=False):
        """
        Evaluate the loss on test data.
        Batch evaluation saves memory and enables this to run on smaller GPUs.
        """
        t_cpu, t_wall = process_time(), time.time()
        predictions = self.predict(out_dict, sess, only_last_layer=only_last_layer)
        if not "loss" in predictions.keys():
            print("Error! Trying to run evaluation, but no labels are provided!")
        loss = predictions["loss"]
        string = 'loss: {:.2e}'.format(loss)
        accuracy, f1 = None, None
        if sess is None:
            string += '\nCPU time: {:.0f}s, wall time: {:.0f}s'.format(process_time()-t_cpu, time.time()-t_wall)
        return string, accuracy, f1, loss

    def fit(self, resume=False, log_device_placement=False, only_last_layer=False, only_hist_step=None):

        if only_last_layer and only_hist_step is not None:
            raise NotImplementedError

        t_cpu, t_wall = process_time(), time.time()

        # if new training:
        if not resume:
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=log_device_placement), graph=self.graph)
            if self.debug:
                sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
            writer = tf.compat.v1.summary.FileWriter(self.get_path('summaries'), self.graph)

            # Initialization
            sess.run(self.op_init)
            glob_step_0 = 0

        # if continuing training:
        else:
            # Save the current state in a zip file
            datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
            filenames = np.asarray(os.listdir(self.get_path('checkpoints')))
            filenames = filenames[np.asarray([(not file.endswith(".zip")) for file in filenames]).astype(bool)]
            print("Zipping current state of the model...")
            with ZipFile(os.path.join(self.get_path('checkpoints'), "zipped_model_" + datetime + '.zip'), 'w') as zipO:
                # Iterate over all the files in directory
                for filename in filenames:
                    # create complete filepath of file in directory
                    filePath = os.path.join(self.get_path('checkpoints'), filename)
                    zipO.write(filePath, arcname=filename)

            # Initialization
            sess = self.get_session()
            if self.debug:
                sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
            writer = tf.compat.v1.summary.FileWriter(self.get_path('summaries'), self.graph)
            glob_step_0 = sess.run(self.global_step)
            print("Resuming training from step", str(glob_step_0) + ".")
        path = os.path.join(self.get_path('checkpoints'), 'model')

        # Set global step
        tf.compat.v1.Variable.load(self.global_step, glob_step_0, sess)

        # Training.
        if self._loss_type == 'cross_entropy':
            accuracies_validation = []
        else:
            accuracies_validation = None
        losses_validation = []
        losses_training = []
        l2_loss_validaton = []
        input_test_dict = self.input_test.vars()
        num_steps = int(self.num_steps)

        if only_hist_step is not None:
            op_train = self.op_train_only_FFs if only_hist_step == 1 else self.op_train_only_hists
        else:
            op_train = self.op_train_only_last if only_last_layer else self.op_train

        try:
            train_range = range(1, num_steps + 1)  # tqdm(range(1, num_steps+1), position=0, leave=True):
            for step in train_range:
                feed_dict = {self.ph_training: True, self.only_last_layer: only_last_layer}
                evaluate = (step % self.eval_frequency == 0) or (step == num_steps)
                if evaluate and self.profile:
                    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                    run_metadata = tf.compat.v1.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None

                if evaluate:
                    learning_rate, loss, label_sums, summary_loc = sess.run([op_train, self.op_loss, self.label_sums, self.op_summary], feed_dict, run_options, run_metadata)
                else:
                    _ = sess.run(op_train, feed_dict, run_options, run_metadata)

                # Periodical evaluation of the model.
                if evaluate:
                    print('step ({} / {}):'.format(step, num_steps))
                    print('  learning_rate = {:.2e}, training loss = {:.2e}'.format(learning_rate, loss))
                    losses_training.append(loss)
                    val_out = dict(zip(input_test_dict.keys(), sess.run(list(input_test_dict.values()))))
                    val_data, val_labels = val_out["data"], val_out["label"]
                    if "var_fracs" in val_out.keys():
                        val_var_fracs = val_out["var_fracs"]
                    if "gce_hist" in val_out.keys():
                        val_gce_hist = val_out["gce_hist"]
                    string, accuracy, f1, loss_val = self.evaluate(val_out, sess, only_last_layer)
                    losses_validation.append(loss_val)
                    print('  validation {}'.format(string))

                    if self._loss_type == "LLH_RATIO":
                        # Summaries for TensorBoard.
                        pred_db_dict = self.predict(val_out, sess=sess, only_last_layer=False, tau_hist=None)
                        logR = pred_db_dict["log_ratio"]
                        mean_logR_cond = np.mean(np.concatenate([logR[:, 0, :], logR[:, 3, :]], axis=0))
                        mean_logR_marg = np.mean(np.concatenate([logR[:, 1, :], logR[:, 2, :]], axis=0))
                        summary = tf.compat.v1.Summary()
                        summary.ParseFromString(summary_loc)
                        summary.value.add(tag='validation/loss', simple_value=loss_val)
                        summary.value.add(tag='validation/mean_cond_logR', simple_value=mean_logR_cond)
                        summary.value.add(tag='validation/mean_marg_logR', simple_value=mean_logR_marg)
                        writer.add_summary(summary, global_step=sess.run(self.global_step))

                        # Calculate posterior for a single map and save
                        post_index = 0
                        im_path = os.path.join(self.get_path("checkpoints"), "Images")
                        mkdir_p(im_path)
                        im_filename = os.path.join(im_path, "posteriors_" + str(sess.run(self.global_step)).zfill(5) + ".pdf")
                        im_filename_FF = os.path.join(im_path, "FF_posteriors_" + str(sess.run(self.global_step)).zfill(5) + ".pdf")

                        # For data generation on-the-fly: only Poissonian models are supported so far.
                        # In this case, prior limits are directly stored as an attribute
                        if hasattr(self, "prior_lims"):
                            prior_scaling = self.prior_lims
                        else:
                            prior_scaling = None

                        # For pre-generated maps: pass PredictionParameters object to plot function
                        if hasattr(self, "pp"):
                            pp = self.pp
                        else:
                            pp = None

                        # Exposure map and template maps
                        exp = template_maps = None
                        if hasattr(self, "template_dict"):
                            exp = self.template_dict["exp"]
                            template_maps = self.template_dict["T_counts"]  # template maps in terms of COUNTS!
                        plot_posterior(self, val_out["data"][post_index, :], n_params=self.ph_labels.shape[1],
                                       n_points_posterior=256, sess=sess, truth=val_out["label"][post_index],
                                       titles=self.pp.parameter_names, filename=im_filename, pp=pp, nside=self.nsides[0], exp=exp,
                                       template_maps=template_maps, filename_FF=im_filename_FF, n_samples_FF=4096,
                                       n_bins_FF=101, plot_z=False)

                    else:
                        if self.aleatoric and (not self.alea_split or only_last_layer):
                            if self.epistemic:
                                pred_db_dict = combine_list_of_dicts([self.predict(val_out, sess=sess, only_last_layer=only_last_layer,
                                                                                   tau_hist=None)
                                                                      for _ in range(self.epi_n_dropout)])
                                pred_db_mc, covar_db_mc = pred_db_dict["logits_mean"], pred_db_dict["covar"]
                                if self.estimate_var_fracs:
                                    pred_var_fracs_mc = pred_db_dict["var_fracs"]
                                    pred_var_fracs = pred_var_fracs_mc.mean(0)
                                mean_aleat_uncertainty = np.sqrt(np.diag(covar_db_mc.mean(1).mean(0)))
                                mean_epist_uncertainty = np.sqrt(np.var(pred_db_mc, 0).mean(0))
                                mean_pred_uncertainty = np.sqrt(mean_epist_uncertainty ** 2 + mean_aleat_uncertainty ** 2)
                                print('  Mean aleatoric uncertainty: ', np.round(mean_aleat_uncertainty, 4))
                                print('  Mean epistemic uncertainty: ', np.round(mean_epist_uncertainty, 4))
                                print('  Mean predictive uncertainty: ', np.round(mean_pred_uncertainty, 4))
                                l2_val_loss = np.mean(np.sum((val_labels - pred_db_mc.mean(0)) ** 2., 1))
                                l1_val_loss = np.mean(np.sum(np.abs(val_labels - pred_db_mc.mean(0)), 1))
                                print('  Dropout propabilities: ', np.round(self.get_dropout_p(sess=sess), 4))

                            else:
                                pred_db_dict = self.predict(val_out, sess=sess, only_last_layer=only_last_layer, tau_hist=None)
                                pred_db = pred_db_dict["logits_mean"]
                                if not self.gaussian_mixture:
                                    covar_db = pred_db_dict["covar"]
                                    mean_aleat_uncertainty = np.sqrt(np.diag(covar_db.mean(0)))
                                    print('  Mean aleatoric uncertainty: ', np.round(mean_aleat_uncertainty, 4))
                                if self.estimate_var_fracs:
                                    pred_var_fracs = pred_db_dict["var_fracs"]
                                l2_val_loss = np.mean(np.sum((val_labels - pred_db) ** 2., 1))
                                l1_val_loss = np.mean(np.sum(np.abs(val_labels - pred_db), 1))

                            l2_loss_validaton.append(l2_val_loss)
                            print('  validation l2 loss: {:.3g}'.format(l2_val_loss))

                        elif not self.aleatoric:
                            pred_db_dict = self.predict(val_out, sess=sess, only_last_layer=only_last_layer, tau_hist=None)
                            pred_db = pred_db_dict["logits_mean"]
                            if self.estimate_var_fracs:
                                pred_var_fracs = pred_db_dict["var_fracs"]
                            l2_val_loss = np.mean(np.sum((val_labels - pred_db) ** 2., 1))
                            l1_val_loss = np.mean(np.sum(np.abs(val_labels - pred_db), 1))

                        if self.estimate_var_fracs:
                            # Weight the loss with true FF of respective template
                            labels_tiled = np.repeat(val_labels, [len(var) for var in self.model_vars], axis=1)
                            labels_tiled_mean = np.mean(labels_tiled, axis=1, keepdims=True)
                            glob_var_frac_val_loss = self.glob_loss_lambda * np.mean(np.mean(labels_tiled * np.abs(val_var_fracs - pred_var_fracs), 1) / labels_tiled_mean)
                            # glob_var_frac_val_loss = np.mean(np.mean(np.abs(val_var_fracs - pred_var_fracs), 1), 0)

                        if self.gce_return_hist:
                            print('  GCE histogram avg. val. l1 loss:', np.abs(val_gce_hist - pred_db_dict["gce_hist"]).mean())

                        print('  CPU time: {:.0f}s, wall time: {:.0f}s'.format(process_time() - t_cpu, time.time() - t_wall))
                        print('   ')

                        # Summaries for TensorBoard.
                        summary = tf.compat.v1.Summary()
                        summary.ParseFromString(summary_loc)
                        if self.aleatoric and (not self.alea_split or only_last_layer) and not self.gaussian_mixture:
                            summary.value.add(tag='validation/loglikelihood', simple_value=-loss_val)
                            summary.value.add(tag='validation/l2', simple_value=l2_val_loss)
                            summary.value.add(tag='validation/l1', simple_value=l1_val_loss)
                            summary.value.add(tag='validation/uncertainties/aleatoric',
                                              simple_value=np.sqrt((mean_aleat_uncertainty ** 2).sum()))
                            if self.epistemic:
                                summary.value.add(tag='validation/uncertainties/epistemic',
                                                  simple_value=np.sqrt((mean_epist_uncertainty ** 2).sum()))
                                summary.value.add(tag='validation/uncertainties/predictive',
                                                  simple_value=np.sqrt((mean_pred_uncertainty ** 2).sum()))
                        elif not self.aleatoric or self.gaussian_mixture:
                            summary.value.add(tag='validation/l2', simple_value=l2_val_loss)
                            summary.value.add(tag='validation/l1', simple_value=l1_val_loss)
                            summary.value.add(tag='validation/loss', simple_value=loss_val)
                        else:
                            summary.value.add(tag='validation/loss', simple_value=loss_val)

                        if self.estimate_var_fracs:
                            summary.value.add(tag='validation/var_fracs', simple_value=glob_var_frac_val_loss)
                        writer.add_summary(summary, global_step=sess.run(self.global_step))

                    if self.profile:
                        writer.add_run_metadata(run_metadata, 'meta_step_{}'.format(sess.run(self.global_step)))

                    # Save model parameters (for evaluation).
                    self.op_saver.save(sess, path, global_step=sess.run(self.global_step))
                    print("Checkpoint", sess.run(self.global_step), "saved.")

        except KeyboardInterrupt:
            print('Optimization stopped by the user')
        if self._loss_type == 'cross_entropy':
            print('validation accuracy: best = {:.2f}, mean = {:.2f}'.format(max(accuracies_validation), np.mean(accuracies_validation[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies_validation, losses_validation, losses_training, t_step

    def get_tensor(self, name, feed_dict=None):
        sess = self.get_session()
        tensor = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(tensor, feed_dict=feed_dict)
        sess.close()
        return val

    def get_dropout_p(self, sess):
        ps = np.array([sess.run(layer_p) for layer_p in tf.compat.v1.get_collection('LAYER_P')])
        return ps

    @staticmethod
    def count_trainable_vars(vars=None):
        total_parameters = 0
        for variable in vars or tf.compat.v1.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print("Trainable variables:", total_parameters)
        return total_parameters

    # Methods to construct the computational graph.

    @staticmethod
    def split_mean_var(output, mean_act, k):
        """
            Format NN output into mean and variances (or in case of Laplace llh: loc. parameter, 2 * scale paramaeter)
            :param output: (batch, k * 2 tensor:
                           k elements: mean, then k elements: variances
            :param mean_act: activation function that will be applied to the mean
            :param k: dimension of regression output
            :return: mean, log variances
        """
        assert output.shape[1] == k * 2, "Aleatoric uncertainty estimation: wrong input shape!"
        mean = mean_act(output[:, :k])
        logvar = output[:, k:]
        return mean, logvar

    @staticmethod
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
        assert output.shape[1] == (k * (k + 3) // 2), "Covariance estimation: wrong input shape!"
        mean = mean_act(output[:, :k])
        var = tf.math.exp(norm_const * output[:, k:2 * k])
        var_mat = tf.math.sqrt(tf.matmul(tf.expand_dims(var, 2), tf.expand_dims(var, 1)))

        # Now, build correlation matrix
        rhos = (1 - eps) * tf.math.tanh(alpha * output[:, 2 * k:])

        for i_row in range(k):
            if i_row == 0:
                rho_mat_temp = tf.expand_dims(tf.concat([tf.zeros((output.shape[0], 1)), rhos[:, :k - 1]], axis=1), 1)
            else:
                lower_ind = i_row * k - i_row * (i_row + 1) // 2
                upper_ind = lower_ind + k - 1 - i_row
                new_row = tf.expand_dims(tf.concat([tf.zeros((output.shape[0], i_row + 1)), rhos[:, lower_ind:upper_ind]], axis=1), 1)
                rho_mat_temp = tf.concat([rho_mat_temp, new_row], axis=1)

        rho_mat = tf.linalg.band_part(rho_mat_temp, num_lower=0, num_upper=k) \
                  + tf.linalg.band_part(tf.linalg.matrix_transpose(rho_mat_temp), num_lower=k, num_upper=0) \
                  + tf.tile(tf.expand_dims(tf.eye(k, k), 0), [output.shape[0], 1, 1])

        return mean, rho_mat * var_mat

    @staticmethod
    def split_gaussian_mixture(output, k, n_gaussians, eps_logalpha=1e-14, eps_logsigma=1e-8, distribution_strategy="diff"):
        """
            Format NN output into means, variances, and alpha
            :param output: (batch, k * n_gaussians * 3 tensor:
                           k * n_gaussians elements: means, log variances, alphas
            :param k: dimension of regression output
            :param n_gaussian: number of Gaussians
            :param eps_logalpha: small number for numerical stability
            :param eps_logsigma: small number for numerical stability: NOTE: making this too large introduces errors!
            :param distribution_strategy: "diff" or "ratio", determines how the mapping to the individual Gaussians is done
            :return: mean, log variances, log alphas
        """
        assert output.shape[1] == k * n_gaussians * 3, "Aleatoric uncertainty estimation: wrong input shape!"
        # Split output
        mu_t_i_raw = tf.reshape(output[:, :k * n_gaussians], [-1, n_gaussians, k])
        # Logvars: model variances with softplus function and take log
        logvars = tf.math.log(eps_logsigma + tf.math.softplus(tf.reshape(output[:, k * n_gaussians:2 * k * n_gaussians], [-1, n_gaussians, k])))
        # logvars = tf.reshape(output[:, k * n_gaussians:2 * k * n_gaussians], [-1, n_gaussians, k]))  # this variant assumes exponential activation
        alphas_raw = tf.reshape(output[:, 2 * k * n_gaussians:], [-1, n_gaussians, k])
        # First: apply softplus to mu_t_i_raw to enforce positivity (shifted ELU works just as well)
        mu_t_i = tf.nn.softplus(mu_t_i_raw)
        # mu_t_i = mu_t_i_raw
        # Apply softmax to alphas over n_gaussians dimension
        alphas_softmax = tf.nn.softmax(alphas_raw, axis=1)
        # Get weighted means (sum_{gaussians} alpha * mu)
        mu_t = tf.reduce_sum(mu_t_i * alphas_softmax, 1, keepdims=True)
        # Apply softmax over template dimension
        mu_t_softmax = tf.nn.softmax(mu_t, -1)
        # Get means of individual Gaussians (cond.: rel. DIFFERENCE between mu's is unaffected!)
        if distribution_strategy == "diff":
            sum_mu_t_i = tf.reduce_sum(mu_t_i, 1, keepdims=True)
            mu_t_i_softmax = mu_t_softmax + (mu_t_i - mu_t) / sum_mu_t_i
        # Get means of individual Gaussians (cond.: RATIO between mu's is unaffected!)
        elif distribution_strategy == "ratio":
            mu_t_i_softmax = mu_t_i * mu_t_softmax / mu_t
        else:
            raise NotImplementedError
        # Calculate log of normalised weights alpha
        logalphas_softmax = tf.math.log(eps_logalpha + alphas_softmax)
        # Return means, log variances, log alpha
        return mu_t_i_softmax, logvars, logalphas_softmax

    def build_graph(self):
        """Build the computational graph of the model."""

        # self.graph = tf.Graph()
        self.graph = self.input_train.data.graph
        with self.graph.as_default():

            # Inputs.
            with tf.compat.v1.name_scope('inputs'):
                self.input_train.data.set_shape([self.batch_size] + list(self.input_train.data.shape[1:]))
                self.input_train.label.set_shape([self.batch_size] + list(self.input_train.label.shape[1:]))
                if hasattr(self.input_train, "var_fracs"):
                    self.input_train.var_fracs.set_shape([self.batch_size, self.input_train.var_fracs.shape[1]])
                    self.ph_var_fracs = self.input_train.var_fracs
                if hasattr(self.input_train, "gce_hist"):
                    self.input_train.gce_hist.set_shape([self.batch_size] + list(self.input_train.gce_hist.shape[1:]))
                    self.ph_gce_hist = self.input_train.gce_hist
                self.ph_data = self.input_train.data
                self.ph_labels = self.input_train.label
                self.ph_training = tf.compat.v1.placeholder(tf.bool, (), 'training')
                self.only_last_layer = tf.compat.v1.placeholder_with_default(False, (), 'only_last_layer')

                # For CDF pinball loss for the histogram: placeholder, which by default gives random quantiles between 0 and 1
                if self.gce_return_hist and self.gce_hist_loss == "CDF_pinball":
                    if self.gce_hist_tau_dist == "uniform":
                        self.ph_tau_hist = tf.compat.v1.placeholder_with_default(
                            tf.random.uniform([self.batch_size, 1], 0.0, 1.0), [self.batch_size, 1], 'tau_hist')
                    elif self.gce_hist_tau_dist == "arcsin":
                        self.ph_tau_hist = tf.compat.v1.placeholder_with_default(
                            0.5 + 0.5 * tf.math.sin(tf.random.uniform([self.batch_size, 1], -np.pi, np.pi)), [self.batch_size, 1], 'tau_hist')
                    else:
                        raise NotImplementedError

            # Model.
            op_logits_dict = self.inference(self.ph_data, self.ph_training)

            if self.gce_return_hist:
                self.op_loss_full = self.loss(op_logits_dict, self.ph_labels, self.regularization)
                self.op_loss = self.op_loss_full[0]  # this is the flux fraction loss

                if self.gce_hist_split:
                    self.op_train_only_FFs = self.training_hist_separate(self.op_loss_full, do_FFs=True, do_hists=False)
                    self.op_train_only_hists = self.training_hist_separate(self.op_loss_full, do_FFs=False, do_hists=True)
                else:
                    do_FFs = False if self.gce_only_hist else True
                    self.op_train = self.training_hist_separate(self.op_loss_full, do_FFs=do_FFs, do_hists=True)

            else:
                if self._loss_type == "LLH_RATIO":
                    self.op_loss = self.loss_llh_ratio(op_logits_dict)
                else:
                    self.op_loss = self.loss(op_logits_dict, self.ph_labels, self.regularization)
                self.op_train = self.training(self.op_loss)

            # Second stage of training (only last layer):
            # Option 1: -> go to full covariance matrix
            if self.aleatoric and self.alea_split:
                self.op_train_only_last = self.training_last_only(self.op_loss)

            self.op_prediction = op_logits_dict

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.compat.v1.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.compat.v1.summary.merge_all()
            self.op_saver = tf.compat.v1.train.Saver(max_to_keep=5)

        self.count_trainable_vars()
        self.graph.finalize()

    def inference(self, data, training):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout and
            batch normalization.
            True: the model is run for training.
            False: the model is run for evaluation.
        """

        if self._loss_type == "LLH_RATIO":
            logits = self._inference_llh_ratio(data, training)
        else:
            logits = self._inference(data, training)
        return logits

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        if self._loss_type =='cross_entropy':
            with tf.compat.v1.name_scope('probabilities'):
                probabilities = tf.nn.softmax(logits)
                return probabilities
        else:
            return None

    def loss(self, logits_dict, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.compat.v1.name_scope('loss'):
            if "logits_mean" in logits_dict.keys():
                logits_mean = logits_dict["logits_mean"]
            if "logvar" in logits_dict.keys():
                logvar = logits_dict["logvar"]
            if "logits_covar" in logits_dict.keys():
                logits_covar = logits_dict["logits_covar"]
            if "var_fracs" in logits_dict.keys():
                logits_glob = logits_dict["var_fracs"]
            if "gce_hist" in logits_dict.keys():
                logits_hist = logits_dict["gce_hist"]

            # Gaussian mixture model (only l2 is allowed):
            if self.gaussian_mixture:
                assert self._loss_type == "l2", "Loss type must be set to l2 for Gaussian mixture model!"
                try:
                    logits_mean_mixture = logits_dict["logits_mean_mixture"]
                    logvar_mixture = logits_dict["logvar_mixture"]
                    logalpha_mixture = logits_dict["logalpha_mixture"]
                except KeyError:
                    raise KeyError("Definition of logits_dict for Gaussian mixture model is not correct!")
                with tf.compat.v1.name_scope('Loss'):
                    HALF_LOG_TWOPI = 0.91893853320467267
                    two_sigma_square = 2.0 * tf.math.exp(logvar_mixture)
                    dist_scaled = (tf.expand_dims(labels, 1) - logits_mean_mixture) ** 2.0 / two_sigma_square  # don't add eps here: already included in two_sigma_square
                    # Truncation term
                    if self.truncate_gaussians:
                        standard_norm = tfd.Normal(0, 1)
                        sigma_inv = tf.math.rsqrt(two_sigma_square / 2.0)
                        trunc_term = - tf.math.log(standard_norm.cdf((1.0 - logits_mean_mixture) * sigma_inv) \
                                                 - standard_norm.cdf((0.0 - logits_mean_mixture) * sigma_inv))
                    else:
                        trunc_term = 0.0
                    # # # # # # #
                    exponent = logalpha_mixture - HALF_LOG_TWOPI - logvar_mixture / 2.0 - dist_scaled + trunc_term
                    fit_loss = - tf.reduce_mean(tf.reduce_sum(tf.math.reduce_logsumexp(exponent, axis=1), -1))
                    tf.compat.v1.summary.scalar('fit_loss', fit_loss)
                    # DEBUG:
                    alpha_sum = tf.reduce_mean(tf.reduce_sum(tf.math.exp(logalpha_mixture), 1))
                    tf.compat.v1.summary.scalar('avg_alpha_sum', alpha_sum)
                    logalpha_loss = - tf.reduce_mean(tf.reduce_mean(tf.math.reduce_logsumexp(logalpha_mixture, axis=1), -1))
                    dist_loss = - tf.reduce_mean(tf.reduce_mean(tf.math.reduce_logsumexp(-dist_scaled, axis=1), -1))
                    logvar_mixture_loss = - tf.reduce_mean(tf.reduce_mean(tf.math.reduce_logsumexp(-logvar_mixture / 2.0, axis=1), -1))
                    tf.compat.v1.summary.scalar('logalpha_loss', logalpha_loss)
                    tf.compat.v1.summary.scalar('dist_loss', dist_loss)
                    tf.compat.v1.summary.scalar('logvar_mixture_loss', logvar_mixture_loss)
                    if self.truncate_gaussians:
                        trunc_loss = - tf.reduce_mean(tf.reduce_mean(tf.math.reduce_logsumexp(trunc_term, axis=1), -1))
                        tf.compat.v1.summary.scalar('trunc_loss', trunc_loss)
                    l2_loss = tf.reduce_mean(tf.reduce_sum((labels - logits_mean) ** 2, 1))
                    l1_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(labels - logits_mean), 1))
                    tf.compat.v1.summary.scalar('l2', l2_loss)
                    tf.compat.v1.summary.scalar('l1', l1_loss)

            # L2 type losses
            elif self._loss_type == 'l2':
                with tf.compat.v1.name_scope('Loss'):
                    if self.aleatoric:
                        l2_loss = tf.reduce_mean(tf.reduce_sum((labels - logits_mean) ** 2, 1))
                        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(labels - logits_mean), 1))
                        tf.compat.v1.summary.scalar('l2', l2_loss)
                        tf.compat.v1.summary.scalar('l1', l1_loss)

                    def aleatoric_loss_l2():
                        return l2_loss

                    # Note: this is modulo a constant
                    def aleatoric_loss_llh():
                        self.logits_mean, self.logits_covar = logits_mean, logits_covar
                        err = np.sqrt(self.covar_scaling) * tf.expand_dims(self.logits_mean - labels, -1)
                        term1 = tf.squeeze(err * tf.linalg.matmul(tf.linalg.inv(self.logits_covar), err), -1)
                        term2 = tf.math.log(np.finfo(np.float32).tiny + tf.linalg.det(self.logits_covar))
                        max_llh_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(term1, 1) + term2) / 2.0
                        return max_llh_loss, term1, term2

                    def aleatoric_loss_llh_only_var():
                        self.logits_mean, self.logvar = logits_mean, logvar
                        err = np.sqrt(self.covar_scaling) * (self.logits_mean - labels)
                        precision = tf.exp(-self.logvar)
                        term1 = err ** 2 * precision
                        term2 = self.logvar
                        max_llh_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(term1 + term2, 1)) / 2.0
                        return max_llh_loss, term1, term2

                    # Set max. llh loss
                    max_llh_loss = (lambda: aleatoric_loss_llh_only_var()[0]) if self.alea_only_var else \
                                   (lambda: aleatoric_loss_llh()[0])
                    max_llh_loss_full = aleatoric_loss_llh_only_var if self.alea_only_var else aleatoric_loss_llh

                    if self.aleatoric and self.alea_split:  # step 1: l2 loss, step 2: max. llh loss
                        fit_loss = tf.cond(self.only_last_layer, true_fn=max_llh_loss, false_fn=aleatoric_loss_l2)
                        tf.compat.v1.summary.scalar('fit_loss', fit_loss)
                    elif self.aleatoric and not self.alea_split:  # don't split it and directly max. llh loss
                        fit_loss, term1, term2 = max_llh_loss_full()
                        tf.compat.v1.summary.scalar('fit_loss', fit_loss)
                        tf.compat.v1.summary.scalar('term1', tf.reduce_mean(tf.reduce_sum(term1, 1)))
                        tf.compat.v1.summary.scalar('term2', tf.reduce_mean(tf.reduce_sum(term2, 1)))
                    else:
                        fit_loss = tf.reduce_mean(tf.reduce_sum((labels - logits_mean) ** 2, 1))
                        tf.compat.v1.summary.scalar('l2', fit_loss)

            # L1 type losses
            elif self._loss_type == 'l1':
                with tf.compat.v1.name_scope('Loss'):
                    if self.aleatoric:
                        l2_loss = tf.reduce_mean(tf.reduce_sum((labels - logits_mean) ** 2, 1))
                        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(labels - logits_mean), 1))
                        tf.compat.v1.summary.scalar('l2', l2_loss)
                        tf.compat.v1.summary.scalar('l1', l1_loss)

                    def aleatoric_loss_l1():
                        return l1_loss

                    def aleatoric_loss_llh():
                        raise NotImplementedError

                    def aleatoric_loss_llh_only_var():
                        location_par, log_twice_scale = logits_mean, logvar
                        err = tf.math.abs(location_par - labels)
                        pre_fac = 2 * tf.exp(-log_twice_scale)  # the 2 is there because the prefactor should be 1 / b = 2 / (2b) = 2 exp(-log(2b))
                        term1 = pre_fac * err
                        term2 = log_twice_scale
                        max_llh_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(term1 + term2, 1))
                        return max_llh_loss, term1, term2

                    # Set max. llh loss
                    max_llh_loss = lambda: aleatoric_loss_llh_only_var()[0] if self.alea_only_var else lambda: \
                                            aleatoric_loss_llh()[0]
                    max_llh_loss_full = aleatoric_loss_llh_only_var if self.alea_only_var else aleatoric_loss_llh

                    if self.aleatoric and self.alea_split:  # step 1: l1 loss, step 2: max. llh loss
                        fit_loss = tf.cond(self.only_last_layer, true_fn=max_llh_loss, false_fn=aleatoric_loss_l1)
                        tf.compat.v1.summary.scalar('fit_loss', fit_loss)
                    elif self.aleatoric and not self.alea_split:  # don't split it and directly max. llh loss
                        fit_loss, term1, term2 = max_llh_loss_full()
                        tf.compat.v1.summary.scalar('fit_loss', fit_loss)
                        tf.compat.v1.summary.scalar('term1', tf.reduce_mean(tf.reduce_sum(term1, 1)))
                        tf.compat.v1.summary.scalar('term2', tf.reduce_mean(tf.reduce_sum(term2, 1)))
                    else:
                        fit_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(labels - logits_mean), 1))
                        tf.compat.v1.summary.scalar('l1', fit_loss)

            # Cross entropy
            elif self._loss_type == "x-ent":
                with tf.compat.v1.name_scope('Loss'):
                    if hasattr(self, "last_act"):
                        assert self.last_act == "linear", "Linear activation must be chosen when using softmax cross-entropy!"
                    logits_softmax = tf.nn.softmax(logits_mean)
                    l2_loss = tf.reduce_mean(tf.reduce_sum((labels - logits_softmax) ** 2, 1))
                    l1_loss = tf.reduce_mean(tf.reduce_sum(tf.math.abs(labels - logits_softmax), 1))
                    tf.compat.v1.summary.scalar('l2', l2_loss)
                    tf.compat.v1.summary.scalar('l1', l1_loss)
                    fit_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_mean, axis=-1))
                    tf.compat.v1.summary.scalar('fit_loss', fit_loss)

            # No loss
            elif self._loss_type.upper() == "NONE":
                fit_loss = tf.constant(0, dtype=tf.float32)

            else:
                raise NotImplementedError

            with tf.compat.v1.name_scope('regularization'):
                n_weights = np.sum(self.regularizers_size)
                regularization *= tf.add_n(self.regularizers) / n_weights

            # Global losses
            if "gce_hist" in logits_dict.keys() and self.gce_return_hist:
                with tf.compat.v1.name_scope('Loss_histogram'):
                    labels_mean = tf.ones(labels.shape[0]) / labels.shape[1]
                    hist_delta = self.ph_gce_hist - logits_hist
                    if self.gce_hist_FF_weights_loss:
                        # Weight the loss with true FF of respective template (shape: n_batch x 1 x n_channels_hist, 1 is for the GCE bin dimension)
                        weighting = tf.expand_dims(tf.transpose(tf.stack([labels[:, hist_ind] / labels_mean for hist_ind in self.gce_hist_indices]), [1, 0]), 1)
                    else:
                        weighting = tf.ones((self.ph_gce_hist.shape[0], 1, self.ph_gce_hist.shape[2]))

                    # Define the loss
                    if self.gce_hist_loss == "l1":
                        model_hist_loss = tf.reduce_mean(weighting * tf.math.abs(hist_delta))  # avg. over n_batch, n_channels_hist, and bins
                    elif self.gce_hist_loss == "l2":
                        model_hist_loss = tf.reduce_mean(weighting * hist_delta ** 2)  # avg. over n_batch, n_channels_hist, and bins
                    elif self.gce_hist_loss == "EM1":
                        model_hist_loss = emd_loss(self.ph_gce_hist, logits_hist, 1, weights=weighting)
                    elif self.gce_hist_loss == "EM2":
                        model_hist_loss = emd_loss(self.ph_gce_hist, logits_hist, 2, weights=weighting)
                    elif self.gce_hist_loss == "CJS":
                        model_hist_loss = cjs_loss(self.ph_gce_hist, logits_hist, weights=tf.squeeze(weighting, 1))
                    elif self.gce_hist_loss == "x-ent":  # NOTE: NO softmax must be applied to the estimated histogram in this case!
                        model_hist_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ph_gce_hist, logits=logits_hist, axis=1) * tf.squeeze(weighting, 1))
                        # model_hist_loss = tf.reduce_mean(tf.losses.binary_crossentropy(self.ph_gce_hist, logits_hist) * labels[:, gce_index] / labels_mean)
                    elif self.gce_hist_loss == "CDF_pinball":
                        model_hist_loss = cdf_quantile_loss(self.ph_gce_hist, logits_hist, self.ph_tau_hist,
                                                            weights=weighting, smoothing=self.gce_hist_pinball_smoothing)
                    else:
                        raise NotImplementedError
                    model_hist_loss *= self.gce_hist_lambda

                    # Additional losses for logging
                    if self.gce_hist_loss != "x-ent":
                        hist_loss_l1 = tf.reduce_mean(weighting * tf.math.abs(hist_delta))
                        hist_loss_l2 = tf.reduce_mean(weighting * hist_delta ** 2)
                        hist_loss_em2 = emd_loss(self.ph_gce_hist, logits_hist, 2, weights=weighting)
                        hist_loss_cjs = cjs_loss(self.ph_gce_hist, logits_hist, weights=tf.squeeze(weighting, 1))
                        tf.compat.v1.summary.scalar('Loss/gce_hist_l1', hist_loss_l1)
                        tf.compat.v1.summary.scalar('Loss/gce_hist_l2', hist_loss_l2)
                        tf.compat.v1.summary.scalar('Loss/gce_hist_em2', hist_loss_em2)
                        tf.compat.v1.summary.scalar('Loss/gce_hist_cjs', hist_loss_cjs)

            if "var_fracs" in logits_dict.keys() and self.estimate_var_fracs:
                with tf.compat.v1.name_scope('Loss_var_frac'):
                    # Weight the loss with true FF of respective template
                    labels_tiled = tf.repeat(labels, [len(var) for var in self.model_vars], axis=1)
                    labels_tiled_mean = tf.reduce_mean(labels_tiled, axis=1, keepdims=True)
                    model_var_loss = self.glob_loss_lambda * tf.reduce_mean(tf.reduce_mean(labels_tiled * tf.math.abs(self.ph_var_fracs - logits_glob), 1) / labels_tiled_mean)

            # Add fit_loss and regularisation losses
            loss = fit_loss + regularization

            # Add global losses: template variant fractions
            # NOTE: model_hist_loss is treated separately, so DON'T ADD HERE!
            if "gce_hist" in logits_dict.keys() and self.gce_return_hist:
                # loss += model_hist_loss
                tf.compat.v1.summary.scalar('Loss/gce_hist', model_hist_loss)

            if "var_fracs" in logits_dict.keys() and self.estimate_var_fracs:
                loss += model_var_loss
                tf.compat.v1.summary.scalar('Loss/var_fracs', model_var_loss)

            # Add concrete dropout regularisation losses
            if self.epistemic:
                concrete_dropout_loss = tf.reduce_sum(tf.compat.v1.losses.get_regularization_losses())
                loss += concrete_dropout_loss
                tf.compat.v1.summary.scalar('concrete_dropout_reg', concrete_dropout_loss)

            # Add lambda penalty term if needed
            if self.lambda_penalty > 0:
                with tf.compat.v1.name_scope('penalty_loss'):
                    self.label_sums = tf.reduce_sum(input_tensor=logits_mean, axis=1)
                    self.penalty_loss = tf.reduce_mean(input_tensor=tf.square(self.label_sums - 1.0))
                loss += self.lambda_penalty * self.penalty_loss

            tf.compat.v1.summary.scalar('Loss/regularization', regularization)
            tf.compat.v1.summary.scalar('Loss/total', loss)
            if self.lambda_penalty > 0:
                tf.compat.v1.summary.scalar('Loss/penalty', self.penalty_loss)

            if "gce_hist" in logits_dict.keys() and self.gce_return_hist:
                return loss, model_hist_loss
            else:
                return loss

    def loss_llh_ratio(self, logits_dict):
        # Evaluate cross-entropy loss
        # loss =
        # -ln( exp(lnL(x_a, z_a))/(1+exp(lnL(x_a, z_a))) )
        # -ln( exp(lnL(x_b, z_b))/(1+exp(lnL(x_b, z_b))) )
        # -ln( 1/(1+exp(lnL(x_a, z_b))) )
        # -ln( 1/(1+exp(lnL(x_b, z_a))) )
        # input shape: n_batch // 2 x 4 (aa ab ba bb) x n_params
        with tf.compat.v1.name_scope('loss'):
            ls = tf.math.log_sigmoid
            logR = logits_dict["log_ratio"]
            loss = - (ls(logR[:, 0, :]) + ls(-logR[:, 1, :]) + ls(-logR[:, 2, :]) + ls(logR[:, 3, :]))
            loss = tf.reduce_sum(loss) / (self.batch_size // 2)
            tf.compat.v1.summary.scalar('llh_loss', loss)
        return loss


    def training(self, loss):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.compat.v1.name_scope('training'):
            # Learning rate.
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = self.scheduler(self.global_step)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            optimizer = self.optimizer(learning_rate)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=self.global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                elif not self.deactivate_hists:
                    tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)
            # Add control dependencies to compute gradients and moving averages (batch normalization).
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([op_gradients] + update_ops):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    def training_last_only(self, loss):
        """Adds to the loss model the Ops required to generate and apply gradients, only for the last FC layer."""
        with tf.compat.v1.name_scope('training'):
            # Learning rate.
            learning_rate = self.scheduler(self.global_step)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            optimizer = self.optimizer(learning_rate)
            var_list = [tf.compat.v1.trainable_variables("logits")]

            grads_last_layer = optimizer.compute_gradients(loss, var_list=var_list)
            op_gradients_last_layer = optimizer.apply_gradients(grads_last_layer, global_step=self.global_step)
            # Histograms.
            for grad, var in grads_last_layer:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                elif not self.deactivate_hists:
                    tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)
            # Add control dependencies to compute gradients and moving averages (batch normalization).
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([op_gradients_last_layer] + update_ops):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    def training_hist_separate(self, loss, do_FFs=True, do_hists=True):
        """Adds to the loss model the Ops required to generate and apply gradients.
           Separately for FF and histogram part """
        if do_FFs and do_hists:
            name_scope = 'training'
            do_global_step_FFs = False
            do_global_step_hists = True
        elif do_FFs:
            name_scope = 'training_FFs'
            assert not self.gce_only_hist, "training_hist_separate was called with do_FFs, but gce_only_hist is set to True!"
            do_global_step_FFs = True
            do_global_step_hists = False
        elif do_hists:
            name_scope = 'training_hists'
            do_global_step_FFs = False
            do_global_step_hists = True
        else:
            raise RuntimeError("training_hist_separate was called with do_FFs and do_hists = False!")

        with tf.compat.v1.name_scope(name_scope):
            loss_FF, loss_hist = loss[0], loss[1]
            # Learning rate.
            # Only create global step if it doesn't exist yet
            if type(self.global_step) == list:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = self.scheduler(self.global_step)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            optimizer = self.optimizer(learning_rate)

            # Get variables that shall be optimized based on loss_FF
            if do_FFs:
                vars_FF = [v for v in tf.compat.v1.trainable_variables() if "FFs" in v.name]
                grads_FF = optimizer.compute_gradients(loss_FF, var_list=vars_FF)
                if do_global_step_FFs:
                    op_gradients_FF = [optimizer.apply_gradients(grads_FF, global_step=self.global_step)]
                else:
                    op_gradients_FF = [optimizer.apply_gradients(grads_FF)]
            else:
                grads_FF = []
                op_gradients_FF = []

            # Get variables that shall be optimized based on loss_hist
            if do_hists:
                vars_hist = [v for v in tf.compat.v1.trainable_variables() if "Hists" in v.name]
                grads_hist = optimizer.compute_gradients(loss_hist, var_list=vars_hist)
                if do_global_step_hists:
                    op_gradients_hist = [optimizer.apply_gradients(grads_hist, global_step=self.global_step)]
                else:
                    op_gradients_hist = [optimizer.apply_gradients(grads_hist)]
            else:
                grads_hist = []
                op_gradients_hist = []

            # Tensorboard histograms
            for grad, var in grads_FF + grads_hist:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                elif not self.deactivate_hists:
                    tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)

            # Add control dependencies to compute gradients and moving averages (batch normalization).
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(op_gradients_FF + op_gradients_hist + update_ops):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.
    def get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def get_session(self, sess=None, checkpoint=None, restore=True):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.compat.v1.Session(graph=self.graph)
            if restore:
                print(self.get_path('checkpoints'))
                if checkpoint is None:
                    filename = tf.train.latest_checkpoint(self.get_path('checkpoints'))
                else:
                    filename = os.path.join(self.get_path('checkpoints'), "model-" + checkpoint)
                self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, stddev=0.1, regularization=True):
        initial = tf.compat.v1.truncated_normal_initializer(0, stddev=stddev)
        var = tf.compat.v1.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var) / stddev**2)
            self.regularizers_size.append(np.prod(shape))
        if not self.deactivate_hists:
            tf.compat.v1.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=False):
        initial = tf.compat.v1.constant_initializer(0)
        # initial = tf.truncated_normal_initializer(0, stddev=1)
        var = tf.compat.v1.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            self.regularizers_size.append(np.prod(shape))
        if not self.deactivate_hists:
            tf.compat.v1.summary.histogram(var.op.name, var)
        return var


# # # # # # # # # # # # GRAPH BASE CLASS # # # # # # # # # # # #
# This class defines the basic operations needed for graph-convolutional NNs
class graphbase(base_model):
    def __init__(self, loss, **kwargs):
        super(graphbase, self).__init__(loss=loss, **kwargs)
        self.filter, self.batch_norm, self.pool, self.activation = None, None, None, None  # these must be set by the subclasses
        self.L, self.K, self.F, self.p, self.M = None, None, None, None, None

    def chebyshev5(self, x, L, Fout, K, do_mc_dropout_if_epistemic=True):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        if np.all(np.asarray(L.shape) > 1):
            lmax = 1.02 * sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]
        else:
            lmax = 1.02
        L = utils.rescale_L(L, lmax=lmax, scale=0.75)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse.reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(a=x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse.sparse_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse.sparse_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        x = tf.transpose(a=x, perm=[1, 3, 2, 0])  # M x N x Fin x K
        x = tf.reshape(x, [M, N, Fin*K])  # M x N x Fin*K
        W = self._weight_variable_cheby(K, Fin, Fout, regularization=True)
        # Concrete dropout
        if self.epistemic and do_mc_dropout_if_epistemic:
            x, W = ConcreteDropout(self.graph, x, W, is_conv_layer=True, weight_regularizer=self.epi_wd, dropout_regularizer=self.epi_dd,
                                   init_min=self.epi_p_init_min, init_max=self.epi_p_init_max).get_input_drop_and_weights()
        x = tf.matmul(x, W)  # M x N x Fout
        return tf.transpose(a=x, perm=[1, 0, 2])  # N x M x Fout

    def _weight_variable_cheby(self, K, Fin, Fout, regularization=True):
        """Xavier like weight initializer for Chebychev coefficients."""
        stddev = 1 / np.sqrt(Fin * (K + 0.5) / 2)
        return self._weight_variable([Fin*K, Fout], stddev=stddev, regularization=regularization)

    def monomials(self, x, L, Fout, K, do_mc_dropout_if_epistemic=False):
        r"""Convolution on graph with monomials."""
        if do_mc_dropout_if_epistemic:
            raise NotImplementedError
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = sparse.csr_matrix(L)
        lmax = 1.02*sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]
        L = utils.rescale_L(L, lmax=lmax)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse.reorder(L)
        # Transform to monomial basis.
        x0 = tf.transpose(a=x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        for k in range(1, K):
            x1 = tf.sparse.sparse_dense_matmul(L, x0)  # M x Fin*N
            x = concat(x, x1)
            x0 = x1
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(a=x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per output feature.
        W = self._weight_variable([Fin*K, Fout], regularization=True)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def bias(self, x):
        """Add one bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return x + b

    def pool_any(self, x, p, i=None, avg=False):
        """Pooling of size p. Should be a power of 2"""
        pool_fct = tf.nn.avg_pool2d if avg else tf.nn.max_pool2d
        if p > 1:
            if hasattr(self, "indexes_extended") and hasattr(self, "indexes") \
                    and hasattr(self, "ind_holes_to_ext") and i is not None:  # general ROI that can have holes: need to go via "indexes_extended" that are a contiguous superset
                smallest_n_contiguous = len(self.indexes_extended[i])
                x_transp = tf.transpose(x, [1, 0, 2])  # M, n_batch, F
                x_zeroes = tf.zeros((smallest_n_contiguous, x.shape[0], x.shape[2]))  # M_extended, n_batch, F
                x_scattered = tf.tensor_scatter_nd_update(x_zeroes, tf.expand_dims(self.ind_holes_to_ext[i], -1), x_transp)
                x_scattered_transp = tf.transpose(x_scattered, [1, 0, 2])  # n_batch, M_ext, F
                x_pooled = pool_fct(input=tf.expand_dims(x_scattered_transp, 3), ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
                x_pooled_3d = tf.squeeze(x_pooled, [3])  # n_batch, M_ext / p, F
                x = tf.gather_nd(x_pooled_3d, tf.tile(tf.expand_dims(tf.expand_dims(self.ind_holes_to_ext[i + 1], 0), -1),
                                         [x.shape[0], 1, 1]), batch_dims=1)
            else:  # assume indices are contiguous (and nested) without holes
                x = tf.expand_dims(x, 3)  # N x M x F x 1
                x = tf.squeeze(tf.nn.max_pool2d(input=x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME'), [3])
            return x
        else:
            return x

    def pool_max(self, x, p, i=None):
        """Max pooling of size p. Should be a power of 2."""
        return self.pool_any(x, p, i, avg=False)

    def pool_average(self, x, p, i=None):
        """Average pooling of size p. Should be a power of 2."""
        return self.pool_any(x, p, i, avg=True)

    def learned_histogram(self, x, bins=20, initial_range=2):
        """A learned histogram layer.

        The center and width of each bin is optimized.
        One histogram is learned per feature map.
        """
        # Shape of x: #samples x #nodes x #features.
        n_features = int(x.get_shape()[2])
        centers = tf.linspace(-float(initial_range), initial_range, bins, name='range')
        centers = tf.expand_dims(centers, axis=1)
        centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
        centers = tf.Variable(
            tf.reshape(tf.transpose(a=centers), shape=[1, 1, n_features, bins]),
            name='centers',
            dtype=tf.float32)
        width = 4 * initial_range / bins  # 50% overlap between bins.
        widths = tf.compat.v1.get_variable(
            name='widths',
            shape=[1, 1, n_features, bins],
            dtype=tf.float32,
            initializer=tf.compat.v1.initializers.constant(value=width, dtype=tf.float32))
        x = tf.expand_dims(x, axis=3)
        # All are rank-4 tensors: samples, nodes, features, bins.
        widths = tf.abs(widths)
        dist = tf.abs(x - centers)
        hist = tf.reduce_mean(input_tensor=tf.nn.relu(1 - dist * widths), axis=1) * (bins/initial_range/4)
        return hist

    def batch_normalization(self, x, training, momentum=0.9):
        """Batch norm layer."""
        # Normalize over all but the last dimension, that is the features.
        return tf.compat.v1.layers.batch_normalization(x, axis=-1, momentum=momentum, epsilon=1e-5, center=False,
                                                       scale=False, training=training)

    def instance_normalization(self, x, eps=1e-8):
        """Instance norm layer"""
        with tf.compat.v1.variable_scope('instance_norm'):
            return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + eps)

    def fc(self, x, Mout, bias=True, do_mc_dropout_if_epistemic=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        # Concrete dropout
        W = self._weight_variable_fc(int(Min), Mout, regularization=True)
        if self.epistemic and do_mc_dropout_if_epistemic:
            x, W = ConcreteDropout(self.graph, x, W, is_conv_layer=False, weight_regularizer=self.epi_wd, dropout_regularizer=self.epi_dd,
                                   init_min=self.epi_p_init_min, init_max=self.epi_p_init_max).get_input_drop_and_weights()
        y = tf.matmul(x, W)
        if bias:
            y += self._bias_variable([Mout], regularization=False)
        return y

    def _weight_variable_fc(self, Min, Mout, regularization=True):
        """Xavier like weight initializer for fully connected layer."""
        stddev = 1 / np.sqrt(Min)
        weight_var = self._weight_variable([Min, Mout], stddev=stddev, regularization=regularization)
        return weight_var

    def _conv_block(self, x, i, training, name_append="", do_mc_dropout_if_epistemic=True):
        """Convolutional block with conv. layer, downscaling, pooling, and activation"""
        with tf.compat.v1.variable_scope('conv{}'.format(i + 1) + name_append):
            L_orig = self.L[i]
            with tf.compat.v1.name_scope('filter'):
                x = self.filter(x, L_orig.copy(), self.F[i], self.K[i], do_mc_dropout_if_epistemic)
                if hasattr(self, "has_se"):
                    if self.has_se[i]:
                        x = self._squeeze_excitation_layer(x, x.shape[-1], 16, "se_" + str(i))
            if i == len(self.p) - 1 and len(self.M) == 0:
                return x  # That is a linear layer before the softmax.
            if self.batch_norm[i] == 1:
                x = self.batch_normalization(x, training)
            elif self.batch_norm[i] == 2:
                x = self.instance_normalization(x)
            x = self.bias(x)
            x = self.activation(x)
            with tf.compat.v1.name_scope('pooling'):
                x = self.pool(x, self.p[i], i)
            return x

    def _res_net_block(self, x, i, training, do_mc_dropout_if_epistemic=True):
        x_orig = x
        L_orig = self.L[i]
        with tf.compat.v1.variable_scope('conv_1_{}'.format(i + 1)):
            with tf.compat.v1.name_scope("filter_"+str(i)+"_1"):
                x = self.filter(x, L_orig.copy(), self.F[i], self.K[i], do_mc_dropout_if_epistemic=do_mc_dropout_if_epistemic)
            x = self.batch_normalization(x, training)
            x = self.activation(x)
        with tf.compat.v1.variable_scope('conv_2_{}'.format(i + 1)):
            with tf.compat.v1.name_scope("filter_"+str(i)+"_2"):
                x = self.filter(x, L_orig.copy(), self.F[i], self.K[i], do_mc_dropout_if_epistemic=do_mc_dropout_if_epistemic)
            x = self.batch_normalization(x, training)
            x = self.activation(x)
        with tf.compat.v1.variable_scope('resnet_{}'.format(i + 1)):
            # If the number of filters changed: do 1 x 1 convolution for x_orig
            if x.shape[-1] != x_orig.shape[-1]:
                x_orig = self.filter(x_orig, L_orig.copy(), self.F[i], int(1), do_mc_dropout_if_epistemic=False)
                if hasattr(self, "has_se"):
                    if self.has_se[i]:
                        x = self._squeeze_excitation_layer(x, x.shape[-1], 16, "se_" + str(i))
            x += x_orig
            x = self.activation(x)
            with tf.name_scope('pooling'):
                x = self.pool(x, self.p[i], i)
            return x

    def _squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.compat.v1.variable_scope(layer_name):
            squeeze = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(input_x)
            with tf.compat.v1.variable_scope("fc1"):
                excitation = self.fc(squeeze, Mout=max(1, out_dim // ratio), bias=True, do_mc_dropout_if_epistemic=False)
                excitation = tf.nn.relu(excitation)
            with tf.compat.v1.variable_scope("fc2"):
                excitation = self.fc(excitation, Mout=out_dim, bias=True, do_mc_dropout_if_epistemic=False)
                excitation = tf.nn.relu(excitation)
                excitation = tf.reshape(excitation, [-1, 1, out_dim])
            with tf.compat.v1.variable_scope("product"):
                scale = input_x * excitation
            return scale


# # # # # # # # # # # # GRAPH CNN # # # # # # # # # # # #
class cgcnn(graphbase):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        batch_norm: apply batch normalization after filtering: 1 True, 0: False, 2: instance norm.
        L: List of Graph Laplacians. Size M x M.
        input_channel: Number of channels of the input image (default 1)

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes for classification.
           Their is no non-linearity for regression...

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.
        loss: loss used to optimize the network
            * 'cross_entropy': loss for classification
            * 'l2': l2 loss 
            * 'l1': l1 loss

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, input_train, input_test, L, F, K, p, batch_norm, M, is_resnet, has_se,
                num_steps, scheduler, optimizer,
                input_channel=1, conv='chebyshev5', pool='max', activation='relu',
                regularization=0.0, dropout=1, batch_size=128, eval_frequency=200,
                dir_name='', profile=False, debug=False, loss='cross_entropy', last_act="sigmoid", lambda_penalty=0.0,
                rel_counts=False, append_tot_counts=False, glob_avg=False, **kwargs):
        super(cgcnn, self).__init__(loss=loss, **kwargs)

        # Forcing label sum towards 1 using lambda only needed if last activation function is NOT softmax
        assert last_act != "softmax" or lambda_penalty == 0, "Set lambda_penalty = 0 when using softmax!"

        # Verify the consistency w.r.t. the number of layers.
        if not len(L) == len(F) == len(K) == len(p) == len(batch_norm) == len(is_resnet) == len(has_se):
            raise ValueError('Wrong specification of the convolutional layers: '
                             'parameters L, F, K, p, batch_norm, is_resnet, has_se must have the same length.')
        if not np.all(np.array(p) >= 1):
            raise ValueError('Down-sampling factors p should be greater or equal to one.')
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        if not np.all(np.mod(p_log2, 1) == 0):
            raise ValueError('Down-sampling factors p should be powers of two.')
        if len(p) > 0:
            if len(M) == 0 and p[-1] != 1:
                raise ValueError('Down-sampling should not be used in the last '
                                 'layer if no fully connected layer follows.')

        if not hasattr(self, "use_templates"):
            self.use_templates = False
            self.template_dict = None

        assert not self.use_templates or np.all(np.asarray(is_resnet) == 0), \
            "Use of templates in combination with ResNet blocks not supported!"

        # Keep the useful Laplacians only. May be zero.
        M_0 = 0 if len(L) == 0 else L[0].shape[0]
        self.L = L

        # Information about NN architecture
        Ngconv = len(p)
        Nfc = len(M)
        train_vars = 0

        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        assert hasattr(self, "use_templates"), "This neural network model does not support the use of templates needed for GCE inference!"
        if self.use_templates and hasattr(self, "template_dict"):
            print("Using", len(self.template_dict), "template(s).")
        M_last = M_0
        for i in range(Ngconv):
            F_last = F[i - 1] if i > 0 else input_channel
            if is_resnet[i]:
                print('  layer {0}: ResNet{0}'.format(i + 1))
                print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i + 1, L[i].shape[0], F[i], p[i], L[i].shape[0] * F[i] // p[i]))
                print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
                train_vars += F_last * F[i] * K[i]
                print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i + 1, F[i], F[i], K[i], F[i] * F[i] * K[i]))
                train_vars += F[i] * F[i] * K[i]
                if F[i] != F_last:
                    print('    shortcut projection weights: F_{0} * F_{1} * 1 = {2} * {3} = {4}'.format(
                    i, i + 1, F_last, F[i], F_last * F[i]))
                    train_vars += F_last * F[i]
            else:
                print('  layer {0}: cgconv{0}'.format(i + 1))
                print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i + 1, L[i].shape[0], F[i], p[i], L[i].shape[0] * F[i] // p[i]))
                print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i + 1, F_last, F[i], K[i], F_last * F[i] * K[i]))
                train_vars += F_last * F[i] * K[i]

            if not (i == Ngconv-1 and len(M) == 0):  # No bias if it's a softmax.
                print('    biases: F_{} = {}'.format(i+1, F[i]))
                train_vars += F[i]
            if batch_norm[i] == 1:
                print('    batch normalization')
            elif batch_norm[i] == 2:
                print('    instance normalization')

        if glob_avg:
            assert Ngconv, "Global average layer needs to follow convolutional layers!"
            M_last = F[-1] // p[-1]
            print("    global average layer")
        else:
            M_last = input_train.data.shape[1] if not Ngconv else L[-1].shape[0] * F[-1] // p[-1]

        for i in range(Nfc):
            name = 'logits' if (i == Nfc-1 and self._loss_type =='cross_entropy') else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            if append_tot_counts and i == 0:
                print("    Additional channel containing log10(total_counts) added.")
                M_last = M_last + 1
            if self.aleatoric and i == Nfc - 1 and not self.alea_only_var:
                channels_out = M[i] * (M[i] + 3) // 2
                print("    Output channels are {0} * ({0} + 3) / 2 = {1} because of prediction of aleatoric covariance "
                      "matrix.".format(M[i], channels_out))
            elif self.aleatoric and i == Nfc - 1 and self.alea_only_var:
                channels_out = M[i] * 2
                print("    Output channels are {0} * 2 = {1} because of prediction of aleatoric uncertainties".format(M[i], channels_out))
            else:
                channels_out = M[i]

            if i == Nfc - 1 and hasattr(self, "model_vars") and self.estimate_var_fracs:
                n_variants = len(flatten_var_fracs(self.model_vars))
                channels_out += n_variants
                print("    ADDITIONALLY: {0} output channels because of template variants!".format(n_variants))

            print('    representation: M_{} = {}'.format(Ngconv+i+1, channels_out))
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, channels_out, channels_out * M_last))
            train_vars += channels_out * M_last
            if i < Nfc - 1:  # No bias if it's a softmax.
                print('    biases: M_{} = {}'.format(Ngconv+i+1, channels_out))
                train_vars += channels_out
            M_last = M[i]

        if self.aleatoric and not self.alea_only_var:
            print("Aleatoric uncertainty covariance matrix will be estimated.")
        elif self.aleatoric and self.alea_only_var:
            print("Aleatoric uncertainties will be estimated.")

        if self.epistemic:
            print("Epistemic uncertainties will be estimated.")
            Nfc_dropout = Nfc if self.epi_dropout_last_layer else Nfc - 1
            print("Number of additional trainable parameters (dropout prob.:) {0} (conv) + {1} (FC) = {2}.".format(len(L), Nfc_dropout, len(L) + Nfc_dropout))
            train_vars += len(L) + Nfc_dropout

        print("The total number of trainable parameters should be", train_vars)
        if self.gce_return_hist:
            print("ADDITIONAL TRAINABLE PARAMETERS FOR GCE HISTOGRAM ESTIMATION!")

        if np.any(has_se):
            print("ADDITIONAL TRAINABLE PARAMETERS FOR SQUEEZE-AND-EXCITATION!")

        if self.gaussian_mixture:
            print("NOTE: Building Gaussian mixture model for non-Gaussian uncertainty estimation! This introduces additional trainable parameters.")

        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.input_channel = input_channel
        self.num_steps = num_steps
        self.scheduler, self.optimizer = scheduler, optimizer
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.batch_norm = batch_norm
        self.dir_name = dir_name
        self.filter = getattr(self, conv)
        self.pool = getattr(self, 'pool_' + pool)
        self.activation = getattr(tf.nn, activation)
        self.profile, self.debug = profile, debug
        self.input_train = input_train
        self.input_test = input_test
        self.last_act = last_act
        self.lambda_penalty = lambda_penalty
        self.rel_counts = rel_counts
        self.append_tot_counts = append_tot_counts
        self.glob_avg = glob_avg
        self.is_resnet = is_resnet
        self.has_se = has_se

        # Build the computational graph.
        self.build_graph()

        # show_all_variables()

    def _inference(self, x, training):

        # If total number of counts is to be appended later: store here
        if self.append_tot_counts:
            tot_counts_log = tf.math.log(tf.reduce_sum(x, axis=1)) / 2.302585092994046  # the hard-coded number is log_e(10)

        # Do relative counts?
        x_abs_counts = x
        x_rel_counts = x / tf.reduce_sum(x, 1, keepdims=True)

        if self.rel_counts:
            x = x_rel_counts
        else:
            x = x_abs_counts

        # Make 3D array
        if len(x.shape) < 3:
            x = tf.expand_dims(x, 2)  # N x M x F=1

        assert hasattr(self, "use_templates"), "This neural network model does not support use of templates!"

        x_input = x

        # Create output dictionary
        logits_dict = dict()

        # If not only histogram is desired: build CNN part for the FF estimation
        if not self.gce_only_hist:
            # Wrap FF estimation in variable scope
            with tf.compat.v1.variable_scope('ConvFFs'):
                for i in range(len(self.p)):
                    if self.is_resnet[i]:
                        x = self._res_net_block(x, i, training)
                    else:
                        x = self._conv_block(x, i, training)

                self._x_before_FC = x

                # Global average layer
                if self.glob_avg:
                    x = tf.expand_dims(tf.keras.layers.GlobalAveragePooling1D()(x), 1)

                # Reshape
                n_samples, n_nodes, n_features = x.get_shape()
                x = tf.reshape(x, [int(n_samples), int(n_nodes * n_features)])

                # Append (log10 of) total number of counts if required
                if self.append_tot_counts and self.rel_counts:
                    x = tf.concat([x, tf.expand_dims(tot_counts_log, -1)], axis=1)

                # Fully connected hidden layers.
                for i, M in enumerate(self.M[:-1]):
                    with tf.compat.v1.variable_scope('fc{}'.format(i + 1)):
                        x = self.fc(x, M)
                        x = self.activation(x)
                        # only do fixed p dropout with fixed rate if not estimating epistemic uncertainty
                        if not self.epistemic and self.dropout < 1.0:
                            dropout = tf.cond(pred=training, true_fn=lambda: float(self.dropout), false_fn=lambda: 1.0)
                            x = tf.nn.dropout(x, 1 - (dropout))

                # Logits linear layer, i.e. softmax without normalization.
                if len(self.M) != 0:
                    with tf.compat.v1.variable_scope('logits'):
                        if self.aleatoric and self.gaussian_mixture:
                            out_dim = self.M[-1] * self.n_gaussians * 3  # n_templates * n_Gaussians * 3 (mean, std, alpha)
                        elif self.aleatoric and not self.alea_only_var:
                            out_dim = self.M[-1] * (self.M[-1] + 3) // 2
                        elif self.aleatoric and self.alea_only_var:
                            out_dim = self.M[-1] * 2
                        else:
                            out_dim = self.M[-1]

                        # Add number of global properties that shall be estimated: template variant fractions
                        if self.estimate_var_fracs:
                            out_dim += self.ph_var_fracs.shape[1]
                        self.second_last_layer = x

                        x = self.fc(x, out_dim, bias=False, do_mc_dropout_if_epistemic=self.epi_dropout_last_layer)

                # Split up into FFs and other global properties (for now, fractions of template variants)
                if self.estimate_var_fracs:
                    x_glob = x[:, -self.ph_var_fracs.shape[1]:]
                    x = x[:, :-self.ph_var_fracs.shape[1]]

                # Set last activation function
                if self.last_act == "sigmoid":
                    last_act = tf.nn.sigmoid
                elif self.last_act == "clip":
                    last_act = lambda val: tf.clip_by_value(val, 0.0, 1.0)
                elif self.last_act == "softmax":
                    last_act = lambda val: tf.nn.softmax(val, axis=-1)
                elif self.last_act == "linear":
                    last_act = lambda val: val
                else:
                    raise NotImplementedError

                # Activation function for the fractions of the template variants
                if self.estimate_var_fracs:
                    with tf.compat.v1.variable_scope('var_fracs_finalizing'):
                        x_glob_stacked = stack_var_fracs(tf.transpose(x_glob, [1, 0]), self.model_vars)
                        x_glob_softmax_array = [None] * len(self.model_vars)
                        for i_temp in range(len(self.model_vars)):
                            x_glob_softmax_array[i_temp] = tf.nn.softmax(x_glob_stacked[i_temp], axis=0)
                        x_glob_out = tf.transpose(tf.concat(x_glob_softmax_array, axis=0), [1, 0])
                        logits_dict["var_fracs"] = x_glob_out

                # if no estimation of aleatoric uncertainty covariance: apply last activation
                if not self.aleatoric:
                    with tf.compat.v1.variable_scope('last_act'):
                        x = last_act(x)
                        logits_dict["logits_mean"] = x

                # if Gaussian mixture model
                elif self.gaussian_mixture:
                    logits_dict["logits_mean_mixture"], logits_dict["logvar_mixture"], logits_dict["logalpha_mixture"] = \
                        self.split_gaussian_mixture(output=x, k=self.M[-1], n_gaussians=self.n_gaussians,
                                                    distribution_strategy=self.distribution_strategy)
                    logits_dict["logits_mean"] = tf.reduce_sum(
                        tf.exp(logits_dict["logalpha_mixture"]) * logits_dict["logits_mean_mixture"], 1)

                # if estimation of full aleatoric uncertainty covariance: calculate mean and covariance matrix
                elif not self.alea_only_var:
                    self.combined_output = tf.identity(x, "combined_output")
                    logits_dict["logits_mean"], logits_dict["logits_covar"] = self.split_mean_cov(output=x,
                                                                                                  mean_act=last_act,
                                                                                                  k=self.M[-1])

                # if only estimating means and variances
                elif self.alea_only_var:
                    logits_dict["logits_mean"], logits_dict["logvar"] = self.split_mean_var(output=x, mean_act=last_act,
                                                                                            k=self.M[-1])
        # If only histogram: set logits_mean to zeros
        else:
            logits_dict["logits_mean"] = tf.zeros_like(self.ph_labels)

        # NOW: Histogram for the PS templates
        if self.gce_return_hist:
            if self.use_templates:

                if self.gce_only_hist:
                    raise RuntimeError("Can't calculate residuals when gce_only_hist is True!")

                # If (aleatoric) uncertainties are provided for the FFs: provide 2 extra channels instead of 1:
                # mean - 1 sigma subtracted, and mean + 1 sigma subtracted
                # if self.aleatoric:
                #     if not self.alea_only_var:
                #         import warnings
                #         warnings.warn("WARNING! For full Gaussian covariance matrix for the flux fraction, only the "
                #                       "means will be passed to the histogram part of the NN!")
                #         FF_input = tf.expand_dims(logits_dict["logits_mean"], 2)
                #     else:
                #         FF_std = tf.math.exp(0.5 * logits_dict["logvar"])
                #         FF_input = tf.stack([logits_dict["logits_mean"] - FF_std, logits_dict["logits_mean"] + FF_std], axis=2)
                # else:
                #     FF_input = tf.expand_dims(logits_dict["logits_mean"], 2)

                FF_input = tf.expand_dims(logits_dict["logits_mean"], 2)

                # From flux fractions, get count fractions!
                # NOTE: with aleat. uncertainties, they are NOT required to sum up to one, and in general do not!
                if self.NN_estimates_CFs:  # if NN estimates count fractions
                    count_fracs = FF_input
                else:  # else: need to convert flux fractions to count fractions
                    FF_sum = tf.reduce_sum(FF_input, 1, keepdims=True)
                    count_fracs_unnorm = FF_input * np.expand_dims(self.template_dict["counts_to_flux_ratio"], [0, 2])
                    count_fracs = count_fracs_unnorm / (tf.reduce_sum(count_fracs_unnorm, axis=1, keepdims=True) / FF_sum)

                # Work with normalised counts or absolute counts?
                if self.gce_hist_rel_counts:
                    x_input_hist = tf.expand_dims(x_rel_counts, 2)
                else:
                    x_input_hist = tf.expand_dims(x_abs_counts, 2)

                # Get counts per template
                if self.remove_exp:  # if exposure is removed by dividing by exp / mean(exp): need to convert to "count" maps
                    count_maps = x_input_hist * np.expand_dims(self.template_dict["rescale"], [0, 2])
                    if self.gce_hist_rel_counts:
                        count_maps = count_maps / tf.reduce_sum(count_maps, 1, keepdims=True)  # normalise again to sum up to 1
                else:  # if count maps are shown to the NN, leave as it is
                    count_maps = x_input_hist

                total_counts = tf.reduce_sum(count_maps, 1, keepdims=True)  # at this point, all count_maps sum up to one if rel_counts
                counts_per_temp = total_counts * count_fracs

                # Get ratio between counts per template and template sum
                T_counts_rescale_fac = counts_per_temp / tf.expand_dims(tf.reduce_sum(self.template_dict["T_counts"].T.astype(np.float32), 0, keepdims=True), 2)

                # Now, compute residual after removing Poissonian template
                n_models = self.template_dict["T_counts"].shape[0]
                n_maps = T_counts_rescale_fac.shape[0]
                n_channels_res = T_counts_rescale_fac.shape[2]
                count_maps_modelled_per_temp = tf.convert_to_tensor([[[T_counts_rescale_fac[mp, tmp, j] * self.template_dict["T_counts"][tmp, :] \
                                                            for tmp in range(n_models)] for mp in range(n_maps)] for j in range(n_channels_res)])  # shape: n_channels_res x n_batch x n_models x n_pix
                count_maps_modelled_per_temp = tf.transpose(count_maps_modelled_per_temp, [1, 2, 3, 0])  # -> n_batch x n_models x n_pix x n_channels_res
                temp_poiss_inds = self.template_dict["temp_indices"]
                count_maps_modelled_Poiss = tf.reduce_sum(tf.gather(count_maps_modelled_per_temp, indices=temp_poiss_inds, axis=1), 1)
                count_maps_residual = count_maps - count_maps_modelled_Poiss  # 1 channel - 2 channels if uncertainties are given

                # if maps shall be in terms of flux: convert again from counts to flux
                if self.remove_exp:
                    x_input_hist_channel_2 = count_maps_residual / np.expand_dims(self.template_dict["rescale"], [0, 2])
                else:
                    x_input_hist_channel_2 = count_maps_residual

                x_input_hist_conc = tf.concat([x_input_hist, x_input_hist_channel_2], axis=2)

                # Store:
                logits_dict["count_maps"] = count_maps
                logits_dict["count_maps_modelled_Poiss"] = count_maps_modelled_Poiss
                logits_dict["count_maps_residual"] = count_maps_residual
                logits_dict["input_hist_channel_2"] = x_input_hist_channel_2

            # if no residual shall be computed: simply pass input map to histogram part of the NN
            else:
                x_input_hist_conc = x_input

            # Wrap histogram estimation in variable scope
            with tf.compat.v1.variable_scope('ConvHists'):
                for i in range(len(self.p)):
                    x_hist = x_input_hist_conc if i == 0 else x_hist
                    if self.is_resnet[i]:
                        x_hist = self._res_net_block(x_hist, i, training, do_mc_dropout_if_epistemic=False)
                    else:
                        x_hist = self._conv_block(x_hist, i, training, do_mc_dropout_if_epistemic=False)

                # Save at this stage
                self._x_hist_before_FC = x_hist

                # Global average layer
                if self.glob_avg:
                    x_hist = tf.expand_dims(tf.keras.layers.GlobalAveragePooling1D()(x_hist), 1)

                # Reshape
                n_samples_hist, n_nodes_hist, n_features_hist = x_hist.get_shape()
                x_hist = tf.reshape(x_hist, [int(n_samples_hist), int(n_nodes_hist * n_features_hist)])

                # Append (log10 of) total number of counts if required
                if self.append_tot_counts and self.gce_hist_rel_counts:
                    x_hist = tf.concat([x_hist, tf.expand_dims(tot_counts_log, -1)], axis=1)

                # If CDF pinball loss for GCE histogram: append tau at this point!
                if self.gce_return_hist and self.gce_hist_loss == "CDF_pinball":
                    x_hist = tf.concat([x_hist, (self.ph_tau_hist - 0.5) * 12], axis=1)  # this scaling is taken from: https://github.com/facebookresearch/SingleModelUncertainty/blob/master/aleatoric/regression/joint_estimation.py

                # Fully connected hidden layers.
                for i, M in enumerate(self.M[:-1]):
                    with tf.compat.v1.variable_scope('fc{}'.format(i+1)):
                        x_hist = self.fc(x_hist, M, do_mc_dropout_if_epistemic=False)
                        x_hist = self.activation(x_hist)
                        # do fixed p dropout with fixed rate?
                        if self.dropout < 1.0:
                            dropout = tf.cond(pred=training, true_fn=lambda: float(self.dropout), false_fn=lambda: 1.0)
                            x_hist = tf.nn.dropout(x_hist, 1 - (dropout))

                # Logits linear layer, i.e. softmax without normalization.
                if len(self.M) != 0:
                    with tf.compat.v1.variable_scope('logits'):
                        out_dim_hist = 0
                        out_dim_hist += np.product(self.ph_gce_hist.shape[1:])  # means only (n_bins * n_channels_hist)
                    self.second_last_layer_hist = x_hist

                    x_hist = self.fc(x_hist, out_dim_hist, bias=False, do_mc_dropout_if_epistemic=False)

                x_hist = tf.reshape(x_hist, self.ph_gce_hist.shape)

                # Activation function for the GCE histogram
                with tf.compat.v1.variable_scope('gce_hist_finalizing'):
                    if self.gce_hist_loss != "x-ent":
                        if self.gce_hist_act == "softmax":
                            logits_dict["gce_hist"] = tf.nn.softmax(x_hist, axis=1)
                        elif self.gce_hist_act == "softplus":
                            logits_dict["gce_hist"] = tf.nn.softplus(x_hist) / tf.reduce_sum(tf.nn.softplus(x_hist), axis=1, keepdims=True)
                        else:
                            raise NotImplementedError
                    else:
                        logits_dict["gce_hist"] = x_hist

        # Return predictions
        return logits_dict

    # Inference function for LLH ratio estimation
    def _inference_llh_ratio(self, x, training):
        # If total number of counts is to be appended later: store here
        if self.append_tot_counts:
            tot_counts_log = tf.math.log(
                tf.reduce_sum(x, axis=1)) / 2.302585092994046  # the hard-coded number is log_e(10)

        # Do relative counts?
        x_abs_counts = x
        x_rel_counts = x / tf.reduce_sum(x, 1, keepdims=True)

        if self.rel_counts:
            x = x_rel_counts
        else:
            x = x_abs_counts

        # Make 3D array
        if len(x.shape) < 3:
            x = tf.expand_dims(x, 2)  # N x M x F=1

        # Bring z into shape NEED TO DO THIS AT THE BEGINNING!
        # (n_batch*2, param-shape)  - repeat twice each sample of z - there are n_batch samples
        # repetition is alternating in first dimension: [a, b, a, b, c, d, c, d, ...]
        n_batch, n_params = self.ph_labels.shape
        assert n_batch % 2 == 0, "There must be an even number of samples in the batch for contrastive learning."
        z = tf.squeeze(tf.reshape(tf.repeat(tf.reshape(self.ph_labels, [n_batch // 2, -1, n_params]), 2, axis=0),
                                [2 * n_batch, -1, n_params]), 1)

        # Scale z to be centered around 0 taking values from -6 to 6
        z = (z - 0.5) * 12

        # Only need to pass x ONCE through the convolutional feature extracting NN part, which is independent of z!
        # -> do NOT repeat x here!

        # Create output dictionary
        logits_dict = dict()

        # Build CNN.
        with tf.compat.v1.variable_scope('ConvFFs'):
            for i in range(len(self.p)):
                if self.is_resnet[i]:
                    x = self._res_net_block(x, i, training)
                else:
                    x = self._conv_block(x, i, training)

            # Global average layer
            if self.glob_avg:
                x = tf.expand_dims(tf.keras.layers.GlobalAveragePooling1D()(x), 1)

            # Reshape
            n_samples, n_nodes, n_features = x.get_shape()
            x = tf.reshape(x, [int(n_samples), int(n_nodes * n_features)])

            # Append (log10 of) total number of counts if required
            if self.append_tot_counts and self.rel_counts:
                x = tf.concat([x, tf.expand_dims(tf.repeat(tot_counts_log, 2), -1)], axis=1)

            # NOW: bring x into shape
            # (n_batch*2, data-shape)  - repeat twice each sample of x - there are n_batch samples
            # repetition pattern in first dimension is: [a, a, b, b, c, c, d, d, ...]
            self._x_before_FC = x
            x = tf.repeat(x, 2, axis=0)

        with tf.compat.v1.variable_scope('ConvFFs_legs'):
            # Append z here (label) TODO: for now: only Poissonian, 1 parameter per template
            n_params = self.ph_labels.shape[1]
            x_z_appended = [None] * n_params
            for i_param in range(n_params):
                x_z_appended[i_param] = tf.concat([x, z[:, i_param:i_param+1]], axis=1)

            # Fully connected hidden layers.
            for i, M in enumerate(self.M[:-1]):
                for i_param in range(n_params):
                    with tf.compat.v1.variable_scope('fc{}_{}'.format(i + 1, i_param)):
                        x_z_appended[i_param] = self.fc(x_z_appended[i_param], M, do_mc_dropout_if_epistemic=False)
                        x_z_appended[i_param] = self.activation(x_z_appended[i_param])
                        # dropout?
                        if self.dropout < 1.0:
                            dropout = tf.cond(pred=training, true_fn=lambda: float(self.dropout),
                                              false_fn=lambda: 1.0)
                            x_z_appended[i_param] = tf.nn.dropout(x_z_appended[i_param], 1 - (dropout))

            # Final linear layer
            if len(self.M) != 0:
                out_dim = 1
                for i_param in range(n_params):
                    with tf.compat.v1.variable_scope('log_ratio_{}'.format(i_param)):
                        x_z_appended[i_param] = self.fc(x_z_appended[i_param], out_dim, bias=False,
                                                        do_mc_dropout_if_epistemic=False)

        # Concatenate. NO activation function here!
        log_ratios = tf.concat(x_z_appended, axis=1)

        # Reshape from 2 * n_batch x n_params, 1 to n_batch / 2 x 4 x n_params
        logits_dict["log_ratio"] = tf.reshape(log_ratios, [n_batch // 2, 4, n_params])

        # Abuse logits_mean to store log_ratio of aa, bb, cc, ... only (used for predicting)
        logR = logits_dict["log_ratio"]
        logits_dict["logits_mean"] = tf.reshape(tf.concat([logR[:, 0, :], logR[:, 3, :]], axis=1), [n_batch, n_params])

        return logits_dict

    def get_filter_coeffs(self, layer, ind_in=None, ind_out=None):
        """Return the Chebyshev filter coefficients of a layer.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        K, Fout = self.K[layer-1], self.F[layer-1]
        trained_weights = self.get_tensor('conv{}/weights/Read/ReadVariableOp'.format(layer))  # Fin*K x Fout
        trained_weights = trained_weights.reshape((-1, K, Fout))
        if layer >= 2:
            Fin = self.F[layer-2]
            assert trained_weights.shape == (Fin, K, Fout)

        # Fin x K x Fout => K x Fout x Fin
        trained_weights = trained_weights.transpose([1, 2, 0])
        if ind_in:
            trained_weights = trained_weights[:, :, ind_in]
        if ind_out:
            trained_weights = trained_weights[:, ind_out, :]
        return trained_weights

    def plot_chebyshev_coeffs(self, layer, ind_in=None, ind_out=None,  ax=None, title='Chebyshev coefficients - layer {}'):
        """Plot the Chebyshev coefficients of a layer.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        title : figure title
        """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        trained_weights = self.get_filter_coeffs(layer, ind_in, ind_out)
        K, Fout, Fin = trained_weights.shape
        ax.plot(trained_weights.reshape((K, Fin*Fout)), '.')
        ax.set_title(title.format(layer))
        return ax

    def get_x_before_FC(self, data, sess=None):
        size = data.shape[0]
        x_before_FC = []
        sess = self.get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_data = np.zeros((self.batch_size, *data.shape[1:]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_training: False}
            batch_x = sess.run(self._x_before_FC, feed_dict)
            x_before_FC.append(batch_x[:end - begin])
        x_before_FC = np.concatenate(x_before_FC)
        return x_before_FC


# # # # # # # # # # # # DEEPSPHERE # # # # # # # # # # # #
class deepsphere(cgcnn):
    """
    Spherical convolutional neural network based on graph CNN

    The following are hyper-parameters of the spherical layers.
    They are lists, which length is equal to the number of gconv layers.
        nsides: NSIDE paramter of the healpix package
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        batch_norm: apply batch norm at the end of the filter (bool vector)

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, input_train, input_test, nsides, model_vars, indexes=None, use_4=False, template_path=None,
                 calculate_residual=False, exp_name=None, new_weighting=False, do_neighbor_scaling=False, glob_loss_lambda=1, **kwargs):

        if len(nsides) > 1:
            L, p = utils.build_laplacians(nsides, indexes=indexes, use_4=use_4, new_weighting=new_weighting,
                                          do_neighbor_scaling=do_neighbor_scaling)
        else:
            L, p = [], []
        self.pygsp_graphs = [None] * len(nsides)

        # if loss is LLH-to-evidence ratio: also need to store templates etc.
        loss_is_LLH_ratio = False
        if "loss" in kwargs:
            if kwargs["loss"] == "LLH_RATIO":
                loss_is_LLH_ratio = True

        if "gce_hist_templates" in kwargs:
            gce_hist_templates = kwargs["gce_hist_templates"]
        else:
            gce_hist_templates = None

        # Templates are for specific template VARIANTS for on-the-fly data generation
        model_vars_flat = np.asarray(model_vars).flatten().tolist()
        if len(model_vars_flat) > len(kwargs["models"]):
            raise NotImplementedError("Only a single template variant is currently supported for LLH ratio estimation!")
        model_input_for_templates = model_vars_flat
        # NOTE: for pre-generated data, model_vars_flat contains "None"s:
        # in this case: kwargs["models"] needs to be passed to store_templates()
        if np.all([m is None for m in model_vars_flat]):
            model_input_for_templates = kwargs["models"]

        if (calculate_residual or loss_is_LLH_ratio) and template_path is not None:
            self.template_dict = dict()
            self.store_templates(template_path, model_input_for_templates, indices=None if indexes is None else indexes[0],
                                 exp_name=exp_name, hist_templates=gce_hist_templates, const_exp=kwargs["const_exp"])
            self.use_templates = True
            if calculate_residual:
                print("Using templates to calculate residuals!")
        else:
            self.use_templates = False

        self.nsides = nsides
        self.new_weighting = new_weighting
        self.label_sums = []
        self.model_vars = model_vars
        self.glob_loss_lambda = glob_loss_lambda
        self.indexes = indexes

        super(deepsphere, self).__init__(input_train=input_train, input_test=input_test, L=L, p=p, **kwargs)

    def get_gsp_filters(self, layer,  ind_in=None, ind_out=None):
        """Get the filter as a pygsp format

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        from pygsp import filters

        trained_weights = self.get_filter_coeffs(layer, ind_in, ind_out)
        nside = self.nsides[layer-1]
        if self.pygsp_graphs[layer-1] is None:
            self.pygsp_graphs[layer-1] = utils.healpix_graph(nside=nside)
            self.pygsp_graphs[layer-1].estimate_lmax()
        return filters.Chebyshev(self.pygsp_graphs[layer-1], trained_weights)

    def plot_filters_spectral(self, layer,  ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter of a special layer in the spectral domain.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """
        import matplotlib.pyplot as plt

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)

        if ax is None:
            ax = plt.gca()
        filters.plot(sum=False, ax=ax, **kwargs)

        return ax

    def plot_filters_section(self, layer,  ind_in=None, ind_out=None, ax=None, **kwargs):
        """Plot the filter section on the sphere

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        ax : axes (optional)
        """
        from . import plot

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)
        fig = plot.plot_filters_section(filters, order=self.K[layer-1], **kwargs)
        return fig

    def plot_filters_gnomonic(self, layer,  ind_in=None, ind_out=None, **kwargs):
        """Plot the filter localization on gnomonic view.

        Parameters
        ----------
        layer : index of the layer (starts with 1).
        ind_in : index(es) of the input filter(s) (default None, all the filters)
        ind_out : index(es) of the output filter(s) (default None, all the filters)
        """
        from . import plot

        filters = self.get_gsp_filters(layer,  ind_in=ind_in, ind_out=ind_out)
        fig = plot.plot_filters_gnomonic(filters, order=self.K[layer-1], **kwargs)

        return fig

    def store_templates(self, template_path, models, const_exp, indices, exp_name="fermidata_exposure", hist_templates=None):
        """
        Store specified templates as a model attribute
        :param template_path: path to the template folder
        :param models: list of template names
        :param const_exp: are the maps generated with constant (Fermi mean) exposure?
        :param indices: indices to use from templates: FROM RING TO NESTED!
        :param exp_name: name of exposure template if exposure correction shall be removed
        :param hist_templates: list of templates names for which histogram will be computed.
                               They will not be modelled away to obtain residual.
        """
        self.template_dict = dict()

        # Get names of the templates to load (might need to remove trailing _PS)
        model_names_without_PS = [model if "PS" not in model else model[:-3] for model in models]

        if hist_templates is None:
            self.template_dict["temp_indices"] = np.arange(len(models))
        else:
            self.template_dict["temp_indices"] = np.argwhere([model not in hist_templates for model in models]).flatten()

        # Get exposure and convert to NEST format
        fermi_exp_full = hp.reorder(np.load(os.path.join(template_path, exp_name + ".npy")), r2n=True)
        fermi_exp_full_mean = fermi_exp_full.mean()
        # Calculate rescale: NOTE: calculated on UNMASKED ROI!!
        fermi_rescale = (fermi_exp_full / fermi_exp_full_mean)[indices]

        # Get relevant indices
        self.template_dict["fermi_exp"] = fermi_exp_full[indices]
        self.template_dict["fermi_rescale"] = fermi_rescale

        if const_exp:
            self.template_dict["exp"] = fermi_exp_full_mean * np.ones_like(fermi_rescale)
            self.template_dict["rescale"] = np.ones_like(self.template_dict["exp"])
        else:
            self.template_dict["exp"] = self.template_dict["fermi_exp"]
            self.template_dict["rescale"] = self.template_dict["fermi_rescale"]

        n_pix_ROI = self.template_dict["fermi_exp"].shape[0]
        n_models = len(models)
        self.template_dict["T_counts"] = np.zeros((n_models, n_pix_ROI))
        self.template_dict["T_flux"] = np.zeros((n_models, n_pix_ROI))
        self.template_dict["counts_to_flux_ratio"] = np.zeros((n_models))

        # Iterate over the templates
        for i_name, name in enumerate(model_names_without_PS):
            temp_counts = hp.reorder(get_template(template_path, name), r2n=True)
            temp_counts = temp_counts[indices]
            # if const_exp: remove exposure correction also for count template
            if const_exp:
                temp_counts /= fermi_rescale
                temp_flux = temp_counts
            else:
                # remove exposure correction to go to "flux space"
                temp_flux = temp_counts / fermi_rescale
            counts_to_flux_ratio = temp_counts.sum() / temp_flux.sum()
            self.template_dict["T_counts"][i_name, :] = temp_counts
            self.template_dict["T_flux"][i_name, :] = temp_flux
            self.template_dict["counts_to_flux_ratio"][i_name] = counts_to_flux_ratio


# # # # # # # # # # # # 2D CNN # # # # # # # # # # # #
class cnn2d(base_model):
    """
    2D convolutional neural network (2D ConvNet)

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of filters.
        K: List of filter shape
        p: Stride for each convolution.
        batch_norm: apply batch normalization after filtering (boolean vector)
        input_shape: Size of the input image
        input_channel: Number of channel of the input image

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        conv: graph convolutional layer, e.g. chebyshev5 or monomials.
        pool: pooling, e.g. max or average.
        activation: non-linearity, e.g. relu, elu, leaky_relu.

    Training parameters:
        num_epochs:     Number of training epochs.
        scheduler:      Learning rate scheduler: function that takes the current step and returns the learning rate.
        optimizer:      Function that takes the learning rate and returns a TF optimizer.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.
        profile:        Whether to profile compute time and memory usage. Needs libcupti in LD_LIBRARY_PATH.
        debug:          Whether the model should be debugged via Tensorboard.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def __init__(self, input_train, input_test, model_vars, F, K, p, batch_norm, M, is_resnet,
                 scheduler, optimizer, input_channel=1, num_steps=10000,
                 pool='max', activation='relu', regularization=0.0, dropout=1, batch_size=128, eval_frequency=200,
                 dir_name='', profile=False, input_shape=None, debug=False, append_tot_counts=False,
                 last_act="sigmoid", lambda_penalty=0.0, pool_fac=None, glob_avg=False,
                 glob_loss_lambda=1.0, **kwargs):
        super(cnn2d, self).__init__(**kwargs)

        # Verify the consistency w.r.t. the number of layers.
        if not len(F) == len(K) == len(p) == len(batch_norm) == len(is_resnet):
            raise ValueError('Wrong specification of the convolutional layers: '
                             'parameters L, F, K, p, batch_norm, is_resnet, must have the same length.')
        if not np.all(np.array(p) >= 1):
            raise ValueError('Down-sampling factors p should be greater or equal to one.')
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        if not np.all(np.mod(p_log2, 1) == 0):
            raise ValueError('Down-sampling factors p should be powers of two.')
        if len(M) == 0 and p[-1] != 1:
            raise ValueError('Down-sampling should not be used in the last '
                             'layer if no fully connected layer follows.')
        assert np.all(np.asarray(batch_norm) <= 1), "Instance norm. for cnn2D not implemented, use batch norm.!"

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: = {}'.format(input_shape))
        nx, ny = input_shape
        for i in range(Ngconv):
            assert p[i] == 1 or pool_fac[i] == 1, \
            "Use EITHER pooling OR stride for downscaling, not both! (Error in conv. " + str(i) + ")."

            nx //= p[i] if p[i] > 1 else pool_fac[i]
            ny //= p[i] if p[i] > 1 else pool_fac[i]

            if is_resnet[i]:
                print('  layer {0}: ResNet{0}'.format(i + 1))
                print('    representation: {0} x {1} x {2} = {3}'.format(nx, ny, F[i], nx * ny * F[i]))
                F_last = F[i - 1] if i > 0 else input_channel
                print('    weights: {0} * {1} * {2} * {3} = {4}'.format(
                    K[i][0], K[i][1], F_last, F[i], F_last * F[i] * K[i][0] * K[i][1]))
                F_last = F[i]
                print('    weights: {0} * {1} * {2} * {3} = {4}'.format(
                    K[i][0], K[i][1], F_last, F[i], F_last * F[i] * K[i][0] * K[i][1]))
            else:
                print('  layer {0}: 2dconv{0}'.format(i + 1))
                print('    representation: {0} x {1} x {2} = {3}'.format(nx, ny, F[i], nx * ny * F[i]))
                F_last = F[i - 1] if i > 0 else input_channel
                print('    weights: {0} * {1} * {2} * {3} = {4}'.format(
                    K[i][0], K[i][1], F_last, F[i], F_last * F[i] * K[i][0] * K[i][1]))

            if not (i == Ngconv - 1 and len(M) == 0):  # No bias if it's a softmax.
                print('    biases: F_{} = {}'.format(i + 1, F[i]))
            if batch_norm[i] == 1:
                print('    batch normalization')
            elif batch_norm[i] == 2:
                print('    instance normalization')

        if glob_avg:
            assert len(F) > 0, "Global average layer needs to follow convolutional layers!"
            M_last = F[-1]
            print("    Global average layer")
        else:
            M_last = nx * ny * input_channel if len(F) == 0 else nx * ny * F[-1]

        for i in range(Nfc):
            name = 'logits' if (i == Nfc - 1 and self._loss_type == 'cross_entropy') else 'fc{}'.format(i + 1)
            print('  layer {}: {}'.format(Ngconv + i + 1, name))
            if append_tot_counts and i == 0:
                print("    Additional channel containing log10(total_counts) added.")
                M_last = M_last + 1
            if self.aleatoric and i == Nfc - 1:
                channels_out = M[i] * (M[i] + 3) // 2
                print("    Output channels are {0} * ({0} + 3) / 2 = {1} because of prediction of aleatoric covariance "
                      "matrix.".format(M[i], channels_out))
            else:
                channels_out = M[i]

            if i == Nfc - 1 and hasattr(self, "model_vars") and self.estimate_var_fracs:
                n_variants = len(flatten_var_fracs(self.model_vars))
                channels_out += n_variants
                print("    ADDITIONALLY: {0} output channels because of template variants!".format(n_variants))

            print('    representation: M_{} = {}'.format(Ngconv + i + 1, channels_out))
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                Ngconv + i, Ngconv + i + 1, M_last, channels_out, channels_out * M_last))
            if i < Nfc - 1:  # No bias if it's a softmax.
                print('    biases: M_{} = {}'.format(Ngconv + i + 1, channels_out))
            M_last = M[i]

        # Store attributes and bind operations.
        self.F, self.K, self.p, self.M, self.is_resnet = F, K, p, M, is_resnet
        self.num_steps = num_steps
        self.scheduler, self.optimizer = scheduler, optimizer
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.batch_norm = batch_norm
        self.dir_name = dir_name
        self.input_train = input_train
        self.input_test = input_test
        self.input_shape = input_shape
        self.input_channel = input_channel
        self.pool = getattr(self, 'pool_' + pool)
        self.pool_fac = pool_fac
        self.activation = getattr(tf.nn, activation)
        self.append_tot_counts = append_tot_counts
        self.last_act = last_act
        self.lambda_penalty = lambda_penalty
        self.profile, self.debug = profile, debug
        self.glob_avg = glob_avg
        self.label_sums = []
        self.model_vars = model_vars
        self.glob_loss_lambda = glob_loss_lambda

        # Build the computational graph.
        self.build_graph()

    def conv2d(self, imgs, nf_out, shape=[5, 5], stride=2, scope="conv2d", regularization=False):
        """Convolutional layer for square images"""

        if not (isinstance(stride, list) or isinstance(stride, tuple)):
            stride = [stride, stride]

        weights_initializer = tf.keras.initializers.glorot_normal()
        # const = tf.keras.initializers.constant()

        with tf.compat.v1.variable_scope(scope):
            sh = [shape[0], shape[1], imgs.get_shape()[-1], nf_out]
            w = tf.compat.v1.get_variable('w', sh, initializer=weights_initializer)
            if regularization:
                self.regularizers.append(tf.nn.l2_loss(w) * np.prod(sh[:-1]))
                self.regularizers_size.append(np.prod(sh))
            conv = tf.nn.conv2d(
                imgs, w, strides=[1, *stride, 1], padding='SAME')

            #             biases = _tf_variable('biases', [nf_out], initializer=const)
            #             conv = tf.nn.bias_add(conv, biases)

            return conv

    def _res_net_block(self, x, nf_out, shape, training, stride=1, pool_fac=1):
        x_orig = x
        x = self.conv2d(x, nf_out, shape, stride=stride, scope="conv2d_1")
        x = self.batch_normalization(x, training)
        x = self.conv2d(x, nf_out, shape, stride=1, scope="conv2d_2")
        x = self.batch_normalization(x, training)
        if stride > 1:
            x_orig = self.pool(x, stride)
        try:
            x += x_orig
        except ValueError:
            print("Error! Number of filters must be constant for ResNet blocks!")
        x = self.bias(x)
        x = self.activation(x)
        if self.pool_fac is not None:
            with tf.name_scope('pooling'):
                x = self.pool(x, pool_fac)
        return x

    def bias(self, x):
        """Add one bias per filter."""
        const = tf.constant_initializer(0.0)
        biases = tf.compat.v1.get_variable('biases', [x.shape[-1]], initializer=const)
        return tf.nn.bias_add(x, biases)

    def pool_max(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.nn.max_pool(x, ksize=[1, p, p, 1], strides=[1, p, p, 1], padding='SAME')
        return x

    def pool_average(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.nn.avg_pool(x, ksize=[1, p, p, 1], strides=[1, p, p, 1], padding='SAME')
        return x

    def learned_histogram(self, x, bins=20, initial_range=2):
        """A learned histogram layer.

        The center and width of each bin is optimized.
        One histogram is learned per feature map.
        """
        # Shape of x: #samples x #nodes x #features.
        n_features = int(x.get_shape()[2])
        centers = tf.linspace(-float(initial_range), initial_range, bins, name='range')
        centers = tf.expand_dims(centers, axis=1)
        centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
        centers = tf.Variable(
            tf.reshape(tf.transpose(centers), shape=[1, 1, n_features, bins]),
            name='centers',
            dtype=tf.float32)
        width = 4 * initial_range / bins  # 50% overlap between bins.
        widths = tf.compat.v1.get_variable(
            name='widths',
            shape=[1, 1, n_features, bins],
            dtype=tf.float32,
            initializer=tf.initializers.constant(value=width, dtype=tf.float32))
        x = tf.expand_dims(x, axis=3)
        # All are rank-4 tensors: samples, nodes, features, bins.
        widths = tf.abs(widths)
        dist = tf.abs(x - centers)
        hist = tf.reduce_mean(tf.nn.relu(1 - dist * widths), axis=1) * (bins / initial_range / 4)
        return hist

    def batch_normalization(self, x, training, momentum=0.9):
        """Batch norm layer."""
        # Normalize over all but the last dimension, that is the features.
        return tf.compat.v1.layers.batch_normalization(x,
                                             axis=-1,
                                             momentum=momentum,
                                             epsilon=1e-5,
                                             center=False,  # Done by bias.
                                             scale=False,  # Done by filters.
                                             training=training)

    def fc(self, x, Mout, bias=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable_fc(int(Min), Mout, regularization=True)
        y = tf.matmul(x, W)
        if bias:
            y += self._bias_variable([Mout], regularization=False)
        return y

    def _weight_variable_fc(self, Min, Mout, regularization=True):
        """Xavier like weight initializer for fully connected layer."""
        stddev = 1 / np.sqrt(Min)
        return self._weight_variable([Min, Mout], stddev=stddev, regularization=regularization)

    def _inference(self, x, training):

        # If total number of counts is to be appended later: store here
        if self.append_tot_counts:
            tot_counts_log = tf.math.log(
                tf.reduce_sum(x, axis=[1, 2])) / 2.302585092994046  # the hard-coded number is log_e(10)
        x = tf.expand_dims(x, 3)

        # Convolutional layers.
        for i in range(len(self.p)):
            # ResNet block
            if self.is_resnet[i]:
                with tf.compat.v1.variable_scope('resnet{}'.format(i + 1)):
                    x = self._res_net_block(x, self.F[i], self.K[i], training=training, stride=self.p[i],
                                            pool_fac=self.pool_fac[i])
            # Normal conv block
            else:
                with tf.compat.v1.variable_scope('conv{}'.format(i + 1)):
                    with tf.name_scope('filter'):
                        x = self.conv2d(x, self.F[i], self.K[i], self.p[i])
                    if i == len(self.p) - 1 and len(self.M) == 0:
                        break  # That is a linear layer before the softmax.
                    if self.batch_norm[i]:
                        x = self.batch_normalization(x, training)
                    x = self.bias(x)
                    x = self.activation(x)
                    if self.pool_fac is not None:
                       with tf.name_scope('pooling'):
                            x = self.pool(x, self.pool_fac[i])

        # Global avg. layer
        if self.glob_avg:
            with tf.compat.v1.variable_scope('GlobAvg'):
                x = tf.expand_dims(tf.expand_dims(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x), 1), 1)

        # Save at this stage
        self._x_before_FC = x

        # Statistical layer (provides invariance to translation and rotation).
        with tf.compat.v1.variable_scope('stat'):
            n_samples, nx, ny, n_features = x.get_shape()
            n_nodes = nx * ny
            x = tf.reshape(x, (n_samples, n_nodes, n_features))
            x = tf.reshape(x, [int(n_samples), int(n_nodes * n_features)])

        # Append (log10 of) total number of counts if required
        if self.append_tot_counts:
            x = tf.concat([x, tf.expand_dims(tot_counts_log, -1)], axis=1)

        # Fully connected hidden layers.
        for i, M in enumerate(self.M[:-1]):
            with tf.compat.v1.variable_scope('fc{}'.format(i + 1)):
                x = self.fc(x, M)
                x = self.activation(x)
                dropout = tf.cond(pred=training, true_fn=lambda: float(self.dropout), false_fn=lambda: 1.0)
                x = tf.nn.dropout(x, 1 - (dropout))

        # Logits linear layer, i.e. softmax without normalization.
        if len(self.M) != 0:
            with tf.compat.v1.variable_scope('logits'):
                out_dim = self.M[-1] * (self.M[-1] + 3) // 2 if self.aleatoric else self.M[-1]

                # Add number of global properties that shall be estimated
                if self.estimate_var_fracs:
                    out_dim += self.ph_var_fracs.shape[1]
                x = self.fc(x, out_dim, bias=False)

        # Split up into FFs and other global properties (for now, fractions of template variants)

        if self.estimate_var_fracs:
            x_glob = x[:, -self.ph_var_fracs.shape[1]:]
            x = x[:, :-self.ph_var_fracs.shape[1]]

        # create output dictionary
        logits_dict = dict()

        # Set last activation function
        if self.last_act == "sigmoid":
            last_act = tf.nn.sigmoid
        elif self.last_act == "clip":
            last_act = lambda val: tf.clip_by_value(val, 0.0, 1.0)
        elif self.last_act == "softmax":
            last_act = lambda val: tf.nn.softmax(val, axis=-1)

        # Activation function for the fractions of the template variants
        if self.estimate_var_fracs:
            with tf.compat.v1.variable_scope('var_fracs_finalizing'):
                x_glob_stacked = stack_var_fracs(tf.transpose(x_glob, [1, 0]), self.model_vars)
                x_glob_softmax_array = [None] * len(self.model_vars)
                for i_temp in range(len(self.model_vars)):
                    x_glob_softmax_array[i_temp] = tf.nn.softmax(x_glob_stacked[i_temp], axis=0)
                x_glob_out = tf.transpose(tf.concat(x_glob_softmax_array, axis=0), [1, 0])
                logits_dict["var_fracs"] = x_glob_out

        # if no estimation of aleatoric uncertainty covariance: apply last activation
        if not self.aleatoric:
            with tf.compat.v1.variable_scope('last_act'):
                x = last_act(x)
                logits_dict["logits_mean"] = x
                return logits_dict

        # if estimation of aleatoric uncertainty covariance: calculate mean and covariance matrix
        else:
            logits_dict["logits_mean"], logits_dict["logits_covar"] = self.split_mean_cov(output=x, mean_act=last_act, k=self.M[-1])
            return logits_dict

    def get_x_before_FC(self, data, sess=None):
        size = data.shape[0]
        x_before_FC = []
        sess = self.get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            batch_data = np.zeros((self.batch_size, *data.shape[1:]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_training: False}
            batch_x = sess.run(self._x_before_FC, feed_dict)
            x_before_FC.append(batch_x[:end - begin])
        x_before_FC = np.concatenate(x_before_FC)
        return x_before_FC


# # # # # # # # # # # # GRAPH U-NET # # # # # # # # # # # #
class graphUNet(graphbase):
    # Init
    def __init__(self, input_train, input_test, nsides, model_vars, glob_M, F_0, F_max, K, batch_norm, M, is_resnet,
                 num_steps, scheduler, optimizer, input_channel=1, conv='chebyshev5', pool='max', regularization=0.0,
                 batch_size=64, eval_frequency=50, dir_name='', profile=False, debug=False, last_act='sigmoid',
                 rel_counts=True, glob_loss_lambda=1, indexes=None, use_4=False, new_weighting=False,
                 do_neighbor_scaling=False, activation_enc='relu', activation_dec='relu', **kwargs):

        if len(nsides) > 1:
            L, p = utils.build_laplacians(nsides, indexes=indexes, use_4=use_4, new_weighting=new_weighting,
                                          do_neighbor_scaling=do_neighbor_scaling)
        else:
            L, p = [], []
        self.pygsp_graphs = [None] * len(nsides)
        self.nsides = nsides
        self.new_weighting = new_weighting
        self.label_sums = []
        self.model_vars = model_vars
        self.glob_M = glob_M
        self.glob_loss_lambda = glob_loss_lambda
        super(graphUNet, self).__init__(input_train=input_train, input_test=input_test, L=L, p=p, **kwargs)

        # Verify the consistency w.r.t. the number of layers.
        if not len(L) == len(p):
            raise ValueError('Wrong specification of the convolutional layers: '
                             'parameters L, p must have the same length.')
        if not np.all(np.array(p) >= 1):
            raise ValueError('Down-sampling factors p should be greater or equal to one.')
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        if not np.all(np.mod(p_log2, 1) == 0):
            raise ValueError('Down-sampling factors p should be powers of two.')

        # Keep the useful Laplacians only. May be zero.
        M_0 = 0 if len(L) == 0 else L[0].shape[0]
        self.L = L

        # Store attributes and bind operations.
        self.L, self.F_0, self.F_max, self.K, self.p, self.M = L, F_0, F_max, K, p, M
        self.activation_enc = getattr(tf.nn, activation_enc)
        self.activation_dec = getattr(tf.nn, activation_dec)
        self.is_resnet = is_resnet
        self.input_channel = input_channel
        self.num_steps = num_steps
        self.scheduler, self.optimizer = scheduler, optimizer
        self.regularization = regularization
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.batch_norm = batch_norm
        self.dir_name = dir_name
        self.filter = getattr(self, conv)
        self.pool = getattr(self, 'pool_' + pool)
        self.activation_enc = getattr(tf.nn, activation_enc)
        self.activation_dec = getattr(tf.nn, activation_dec)
        self.profile, self.debug = profile, debug
        self.input_train = input_train
        self.input_test = input_test
        self.last_act = last_act
        self.rel_counts = rel_counts
        self.is_resnet = is_resnet

        # Build the computational graph.
        self.build_graph()

    # Fit
    def fit(self, resume=False, log_device_placement=False, only_last_layer=False):
        t_cpu, t_wall = process_time(), time.time()
        # if new training:
        if not resume:
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=log_device_placement), graph=self.graph)
            if self.debug:
                sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
            writer = tf.compat.v1.summary.FileWriter(self.get_path('summaries'), self.graph)

            # Initialization
            sess.run(self.op_init)
            glob_step_0 = 0

        # if continuing training:
        else:
            # Save the current state in a zip file
            datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
            filenames = np.asarray(os.listdir(self.get_path('checkpoints')))
            filenames = filenames[np.asarray([(not file.endswith(".zip")) for file in filenames]).astype(bool)]
            print("Zipping current state of the model...")
            with ZipFile(os.path.join(self.get_path('checkpoints'), "zipped_model_" + datetime + '.zip'), 'w') as zipO:
                # Iterate over all the files in directory
                for filename in filenames:
                    # create complete filepath of file in directory
                    filePath = os.path.join(self.get_path('checkpoints'), filename)
                    zipO.write(filePath, arcname=filename)

            # Initialization
            sess = self.get_session()
            if self.debug:
                sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
            writer = tf.compat.v1.summary.FileWriter(self.get_path('summaries'), self.graph)
            glob_step_0 = sess.run(self.global_step)
            print("Resuming training from step", glob_step_0, ".")
        path = os.path.join(self.get_path('checkpoints'), 'model')

        # Set global step
        tf.compat.v1.Variable.load(self.global_step, glob_step_0, sess)

        # Training.
        if self._loss_type == 'cross_entropy':
            accuracies_validation = []
        else:
            accuracies_validation = None
        losses_validation = []
        losses_training = []

        num_steps = int(self.num_steps)

        op_train = self.op_train_only_last if only_last_layer else self.op_train

        try:
            for step in range(1, num_steps+1):
                feed_dict = {self.ph_training: True, self.only_last_layer: only_last_layer}
                evaluate = (step % self.eval_frequency == 0) or (step == num_steps)
                if evaluate and self.profile:
                    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                    run_metadata = tf.compat.v1.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None

                if evaluate:
                    learning_rate, loss, label_sums, summary_loc = sess.run([op_train, self.op_loss, self.label_sums, self.op_summary], feed_dict, run_options, run_metadata)
                else:
                    _ = sess.run(op_train, feed_dict, run_options, run_metadata)

                # Periodical evaluation of the model.
                if evaluate:
                    print('\nstep ({} / {}):'.format(step, num_steps))
                    print('  learning_rate = {:.2e}, training loss = {:.2e}'.format(learning_rate, loss))
                    losses_training.append(loss)
                    val_data, val_labels, val_var_fracs = sess.run([self.input_test.data, self.input_test.label, self.input_test.var_fracs])
                    string, accuracy, f1, loss_val = self.evaluate(val_data, val_labels, val_var_fracs, sess, only_last_layer)
                    losses_validation.append(loss_val)
                    print('  validation {}'.format(string))

                    # Tensorboard: log losses
                    pred_db_dict = self.predict(val_data, sess=sess, only_last_layer=only_last_layer)
                    pred_db = pred_db_dict["logits_mean"]
                    if self.estimate_var_fracs:
                        pred_var_fracs = pred_db_dict["var_fracs"]
                    eps = 1e-8
                    data_exp = np.expand_dims(val_data, -1)
                    relevant_pixels = (data_exp > 0).astype(np.float32)  # pixels with at least 1 count
                    pred_counts = data_exp * pred_db  # predicted counts: FFs predicted by NN * input map
                    scale_fac_l1 = 1 / (np.sqrt(data_exp + eps))
                    scale_fac_l2 = 1 / (data_exp + eps)
                    l2_val_loss = np.mean(np.mean(np.mean((val_labels - pred_counts) ** 2 * scale_fac_l2 * relevant_pixels, 1), -1))
                    l1_val_loss = np.mean(np.mean(np.mean(np.abs(val_labels - pred_counts) * scale_fac_l1 * relevant_pixels, 1), -1))
                    if self.estimate_var_fracs:
                        # Weight the loss with true COUNT FRACTION of respective template
                        label_count_sum = np.sum(val_labels, 1)
                        labels_global = label_count_sum / np.sum(label_count_sum, 1, keepdims=True)
                        labels_tiled = np.repeat(labels_global, [len(var) for var in self.model_vars], axis=1)
                        labels_tiled_mean = np.mean(labels_tiled, axis=1, keepdims=True)
                        glob_var_frac_val_loss = self.glob_loss_lambda * np.mean(np.mean(labels_tiled * np.abs(val_var_fracs - pred_var_fracs), 1) / labels_tiled_mean)
                        # glob_var_frac_val_loss = np.mean(np.mean(np.abs(val_var_fracs - pred_var_fracs), 1), 0)

                    print('  CPU time: {:.0f}s, wall time: {:.0f}s'.format(process_time() - t_cpu, time.time() - t_wall))

                    # Summaries for TensorBoard.
                    summary = tf.compat.v1.Summary()
                    summary.ParseFromString(summary_loc)
                    summary.value.add(tag='validation/l2', simple_value=l2_val_loss)
                    summary.value.add(tag='validation/l1', simple_value=l1_val_loss)
                    if self.estimate_var_fracs:
                        summary.value.add(tag='validation/var_fracs', simple_value=glob_var_frac_val_loss)
                    summary.value.add(tag='validation/loss', simple_value=loss_val)
                    writer.add_summary(summary, global_step=sess.run(self.global_step))
                    if self.profile:
                        writer.add_run_metadata(run_metadata, 'step{}'.format(self.global_step))

                    # Save model parameters (for evaluation).
                    self.op_saver.save(sess, path, global_step=sess.run(self.global_step))
                    print("  Checkpoint", sess.run(self.global_step), "saved.\n")

        except KeyboardInterrupt:
            print('Optimization stopped by the user')
        if self._loss_type == 'cross_entropy':
            print('validation accuracy: best = {:.2f}, mean = {:.2f}'.format(max(accuracies_validation), np.mean(accuracies_validation[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies_validation, losses_validation, losses_training, t_step

    # Build graph
    def build_graph(self):
        """Build the computational graph of the model."""

        # self.graph = tf.Graph()
        self.graph = self.input_train.data.graph
        with self.graph.as_default():

            # Inputs.
            with tf.compat.v1.name_scope('inputs'):
                self.input_train.data.set_shape([self.batch_size, self.input_train.data.shape[1]])
                self.input_train.label.set_shape([self.batch_size] + list(self.input_train.label.shape[1:]))
                self.input_train.var_fracs.set_shape([self.batch_size, self.input_train.var_fracs.shape[1]])
                self.ph_data = self.input_train.data
                self.ph_labels = self.input_train.label
                self.ph_var_fracs = self.input_train.var_fracs
                self.ph_training = tf.compat.v1.placeholder(tf.bool, (), 'training')
                self.only_last_layer = tf.compat.v1.placeholder_with_default(False, (), 'only_last_layer')

            # Model.
            if self.aleatoric:
                raise NotImplementedError
            else:
                op_logits_dict = self.inference(self.ph_data, self.ph_training)
                self.op_loss = self.loss(op_logits_dict, self.ph_labels, self.ph_data, self.regularization)

                self.op_train = self.training(self.op_loss)
                if self.aleatoric and self.alea_split:
                    self.op_train_only_last = self.training_last_only(self.op_loss)

                self.op_prediction = op_logits_dict

                # Initialize variables, i.e. weights and biases.
                self.op_init = tf.compat.v1.global_variables_initializer()

                # Summaries for TensorBoard and Save for model parameters.
                self.op_summary = tf.compat.v1.summary.merge_all()
                self.op_saver = tf.compat.v1.train.Saver(max_to_keep=5)

            self.count_trainable_vars()
            self.graph.finalize()

    # Loss
    def loss(self, logits_dict, labels, data, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.compat.v1.name_scope('loss'):
            logits_mean = logits_dict["logits_mean"]
            if "var_fracs" in logits_dict.keys() and self.estimate_var_fracs:
                logits_glob = logits_dict["var_fracs"]
            with tf.compat.v1.name_scope('Loss_loc'):
                # Local losses
                eps = 1e-8
                data_exp = tf.expand_dims(data, -1)
                relevant_pixels = tf.cast(data_exp > 0, tf.float32)  # pixels with at least 1 count
                pred_counts = data_exp * logits_mean  # predicted counts: FFs predicted by NN * input map
                scale_fac_l1 = tf.math.rsqrt(data_exp + eps)
                scale_fac_l2 = 1 / (data_exp + eps)
                # scale_fac_l1 = scale_fac_l2 = 1.0
                l2_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean((labels - pred_counts) ** 2 * scale_fac_l2 * relevant_pixels, 1), -1))
                l1_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.math.abs(labels - pred_counts) * scale_fac_l1 * relevant_pixels, 1), -1))

                tf.compat.v1.summary.scalar('l2', l2_loss)
                tf.compat.v1.summary.scalar('l1', l1_loss)
                fit_loss = l1_loss if self._loss_type == "l1" else l2_loss

            # Global losses
            if self.model_vars is not None and self.estimate_var_fracs:
                with tf.compat.v1.name_scope('Loss_glob'):
                    # Weight the loss with true COUNT FRACTION of respective template
                    # NOTE: For U-Net, need to calculate global COUNT FRACTION first!
                    label_count_sum = tf.reduce_sum(labels, 1)
                    labels_global = label_count_sum / tf.reduce_sum(label_count_sum, 1, keepdims=True)
                    labels_tiled = tf.repeat(labels_global, [len(var) for var in self.model_vars], axis=1)
                    labels_tiled_mean = tf.reduce_mean(labels_tiled, axis=1, keepdims=True)
                    model_var_loss = self.glob_loss_lambda * tf.reduce_mean(tf.reduce_mean(labels_tiled * tf.math.abs(self.ph_var_fracs - logits_glob), 1) / labels_tiled_mean)

            with tf.compat.v1.name_scope('regularization'):
                n_weights = np.sum(self.regularizers_size)
                regularization *= tf.add_n(self.regularizers) / n_weights

            # Add fit_loss and regularisation losses
            loss = fit_loss + regularization

            # Add global loss
            if self.model_vars is not None and self.estimate_var_fracs:
                loss += model_var_loss

            tf.compat.v1.summary.scalar('Loss/regularization', regularization)
            if self.model_vars is not None and self.estimate_var_fracs:
                tf.compat.v1.summary.scalar('Loss/var_fracs', model_var_loss)
            tf.compat.v1.summary.scalar('Loss/total', loss)
            return loss

    # Upsampling block
    def _upsample(self, x, upsampling_fac=4):
        with tf.name_scope('upsampling'):
            # x_out = tf.keras.layers.UpSampling1D(upsampling_fac)(x)
            # x_out = tf.keras.backend.repeat_elements(x, upsampling_fac, axis=1)
            x_temp_1 = tf.tile(tf.transpose(tf.expand_dims(x, -1), [0, 2, 3, 1]), [1, 1, 1, upsampling_fac])
            x_temp_2 = tf.transpose(tf.reshape(x_temp_1, [x_temp_1.shape[0], x_temp_1.shape[1], -1, upsampling_fac]), [0, 1, 3, 2])
            x_out = tf.transpose(tf.reshape(x_temp_2, [x_temp_2.shape[0], x_temp_2.shape[1], -1]), [0, 2, 1])
        return x_out

    # Inference
    def _inference(self, x, training):

        # Do relative counts?
        if self.rel_counts:
            x = x / tf.reduce_sum(x, 1, keepdims=True)

        # Make 3D array
        if len(x.shape) < 3:
            x = tf.expand_dims(x, 2)  # N x M x F=1

        # Build U-Net
        n_levels = int(np.round(np.log2(x.shape[1]) // 2)) + 1
        x_enc = [None] * n_levels
        x_enc_full = [None] * (n_levels - 1)
        x_dec = [None] * n_levels

        F0 = self.F_0
        F_max = self.F_max

        # Define normalisation
        if self.batch_norm == 1:
            norm_operation = lambda x_: self.batch_normalization(x_, training)
        elif self.batch_norm == 2:
            norm_operation = lambda x_: self.instance_normalization(x_)
        else:
            norm_operation = lambda x_: x_

        # Encoding path
        for i_level in range(n_levels - 1):
            F_level = min(F0 * 2 ** i_level, F_max)
            L = self.L[i_level].copy()  # NOTE: COPY! otherwise, values of Laplacian change!
            with tf.compat.v1.variable_scope("Enc_" + str(i_level) + "_1"):
                if i_level == 0:
                    x_enc[0] = x

                # Block 1
                x_temp = self.filter(x_enc[i_level], L, F_level, self.K, do_mc_dropout_if_epistemic=False)
                x_temp = norm_operation(x_temp)
                x_temp = self.bias(x_temp)
                x_temp = self.activation_enc(x_temp)

            L = self.L[i_level].copy()
            with tf.compat.v1.variable_scope("Enc_" + str(i_level) + "_2"):
                # Block 2
                x_temp = self.filter(x_temp, L, F_level, self.K, do_mc_dropout_if_epistemic=False)
                x_temp = norm_operation(x_temp)
                x_temp = self.bias(x_temp)
                x_temp = self.activation_enc(x_temp)
                x_enc_full[i_level] = x_temp

                if self.is_resnet:
                    with tf.compat.v1.variable_scope("ResNet_" + str(i_level)):
                        L_resnet = self.L[i_level].copy()
                        conv_resnet = self.filter(x_enc[i_level], L_resnet, F_level, 1, do_mc_dropout_if_epistemic=False)
                        x_enc_full[i_level] = self.activation_enc(conv_resnet + x_enc_full[i_level])

                x_enc[i_level + 1] = self.pool(x_enc_full[i_level], self.p[i_level])

        # Decoding path
        x_dec[n_levels - 1] = x_enc[n_levels - 1]
        for i_level in range(n_levels - 1, -1, -1):
            F_level = min(F0 * 2 ** i_level, F_max)
            L = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1), dtype=np.float32).copy() if i_level == n_levels - 1 \
                else self.L[i_level].copy()
            with tf.compat.v1.variable_scope("Dec_" + str(i_level) + "_1"):
                if i_level < n_levels - 1:
                    L_for_reduction = self.L[i_level].copy()
                    with tf.compat.v1.variable_scope("Dec_" + str(i_level) + "_red"):
                        x_dec_reduced = self.filter(x_dec[i_level], L_for_reduction, F_level, self.K, do_mc_dropout_if_epistemic=False)
                        x_temp = tf.concat([x_dec_reduced, x_enc_full[i_level]], axis=2)
                        # x_temp = x_dec_reduced + x_enc_full[i_level]  # sum instead of concat gives somewhat worse results
                else:
                    x_temp = tf.identity(x_dec[i_level], "dec_start")

                if self.is_resnet:
                    x_orig = x_temp

                # Block 1
                x_temp = self.filter(x_temp, L, F_level, self.K, do_mc_dropout_if_epistemic=False)
                x_temp = norm_operation(x_temp)
                x_temp = self.bias(x_temp)
                x_temp = self.activation_dec(x_temp)

            L = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1), dtype=np.float32).copy() if i_level == n_levels - 1 \
                else self.L[i_level].copy()
            with tf.compat.v1.variable_scope("Dec_" + str(i_level) + "_2"):
                # Block 2
                x_temp = self.filter(x_temp, L, F_level, self.K, do_mc_dropout_if_epistemic=False)
                x_temp = norm_operation(x_temp)
                x_temp = self.bias(x_temp)
                x_temp = self.activation_dec(x_temp)

                if self.is_resnet:
                    with tf.compat.v1.variable_scope("ResNet_" + str(i_level)):
                        L_resnet = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, 1),
                                              dtype=np.float32).copy() if i_level == n_levels - 1 \
                            else self.L[i_level].copy()
                        conv_resnet = self.filter(x_orig, L_resnet, F_level, 1, do_mc_dropout_if_epistemic=False)
                        x_temp = self.activation_enc(conv_resnet + x_temp)

                if i_level == n_levels - 1:
                    x_final_bottleneck = tf.identity(x_temp, "final_bottleneck")

                if i_level > 0:
                    with tf.compat.v1.name_scope("Upsample_" + str(i_level)):
                        x_dec[i_level - 1] = self._upsample(x_temp, self.p[i_level - 1])
                else:
                    with tf.compat.v1.name_scope("Pre-out"):
                        x_pre_out = tf.identity(x_temp, "pre_out")

        # create output dictionary
        logits_dict = dict()

        # Fork off FC layers from final bottleneck layer for the estimation of global properties!
        if self.model_vars is not None and self.estimate_var_fracs:
            with tf.compat.v1.variable_scope("Global_branch"):
                # Fully connected hidden layers.
                x_glob = tf.reshape(x_final_bottleneck, [x_final_bottleneck.shape[0], x_final_bottleneck.shape[2]])
                for i, M in enumerate(self.glob_M):
                    with tf.compat.v1.variable_scope('fc{}'.format(i + 1)):
                        x_glob = self.fc(x_glob, M)
                        x_glob = self.activation_dec(x_glob)

                # Logits linear layer, i.e. softmax without normalization.
                with tf.compat.v1.variable_scope('fc_last'):
                    x_glob = self.fc(x_glob, self.ph_var_fracs.shape[1], bias=False)
                    x_glob_stacked = stack_var_fracs(tf.transpose(x_glob, [1, 0]), self.model_vars)
                    x_glob_softmax_array = [None] * len(self.model_vars)
                    for i_temp in range(len(self.model_vars)):
                        x_glob_softmax_array[i_temp] = tf.nn.softmax(x_glob_stacked[i_temp], axis=0)
                    x_glob_out = tf.transpose(tf.concat(x_glob_softmax_array, axis=0), [1, 0])
                    logits_dict["var_fracs"] = x_glob_out

        with tf.compat.v1.variable_scope("Finalizing"):
            # Final convolution to number of output channels
            L = self.L[0].copy()
            out_dim = self.ph_labels.shape[2]
            x_channels_out = self.filter(x_pre_out, L, out_dim, self.K, do_mc_dropout_if_epistemic=False)

            # Set last activation function
            if self.last_act == "sigmoid":
                last_act = tf.nn.sigmoid
            elif self.last_act == "clip":
                last_act = lambda val: tf.clip_by_value(val, 0.0, 1.0)
            elif self.last_act == "softmax":
                last_act = lambda val: tf.nn.softmax(val, axis=-1)
            elif self.last_act == "linear":
                last_act = lambda val: val
            else:
                raise NotImplementedError

            # TODO: implement uncertainty estimation
            # if no estimation of aleatoric uncertainty covariance: apply last activation
            if not self.aleatoric:
                with tf.compat.v1.variable_scope('last_act'):
                    logits_dict["logits_mean"] = last_act(x_channels_out)
                    return logits_dict