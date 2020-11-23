"""
Galactic Centre Excess Neural Network Class
"""
from gce_utils import *
from gce_data import PairGeneratorCNNPreGenerated, PairGeneratorCNNOnTheFly, PairGeneratorUNet, DatasetCNN, DatasetUNet
import shutil
import os
import sys
from copy import copy
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import tensorflow as tf
from deepsphere import models, plot
from pprint import pprint
import dill
import cloudpickle
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'rocket'
print("\n\nFound TF", tf.__version__, ".")
tf.compat.v1.disable_eager_execution()


class GCE_NN():
    def __init__(self, NN_type, HPC=False, path_to_load=None):
        """
        Initialise the GCE NN
        :param NN_type:
                        * "CNN":      use a CNN with maps being generated "on-the-fly" during the training,
                                      except for (potential) point-source emission, for which the maps need
                                      to be generated beforehand.
                                      -> parameter file: parameters_CNN.py
                        * "CNN_pre":  use a CNN with pre-generated maps.
                                      -> parameter file: parameters_CNN_pre_gen.py
                        * "U-Net":    use a U-Net for pixel-wise regression (EXPERIMENTAL!)
                                      -> parameter file: parameters_UNet.py
        :param HPC: running on a supercomputer / GPU environment?
        :param path_to_load: path to a checkpoint folder if the parameters of a trained NN shall be loaded,
                             otherwise, set to None
        """

        # Read the parameters
        self.path_to_load = path_to_load
        self.NN_type = NN_type
        self.HPC = HPC
        self.parse_params()
        self.params_orig = copy(self.__dict__)
        for key in ["path_to_load", "NN_type", "HPC"]:
            del self.params_orig[key]

        # Define the attributes that are to be defined later
        self.generator = None
        self.generator_test = None
        self.ds = None
        self.ds_test = None
        self.input_train = None
        self.input_test = None
        self.input_test_db = None
        self.input_train_dict = None
        self.fermi_counts = None
        self.model = None

    def __getitem__(self, item):
        """Allow accessing class like a dictionary"""
        return getattr(self, item)

    def keys(self):
        """Allow accessing attributes as keys"""
        return self.__dict__.keys()

    def __setitem__(self, key, value):
        """Allow setting attributes like in a dictionary"""
        setattr(self, key, value)

    def parse_params(self):
        """Parse the parameters for the selected NN type"""
        # Load from pickle file
        if self.path_to_load is not None:
            params = load_params_from_pickle(self.path_to_load)
            print("  Loaded NN parameters from", "'" + self.path_to_load + "'.")
            assert self.NN_type == params["NN_type"], "It seems like something is wrong with the NN type... aborting!"

        # Load from parameter file
        else:
            if self.NN_type == "CNN":
                from parameters_CNN import get_params_CNN
                params = get_params_CNN(self.HPC)
            elif self.NN_type == "CNN_pre":
                from parameters_CNN_pre_gen import get_params_CNN_pre_gen
                params = get_params_CNN_pre_gen(self.HPC)
            elif self.NN_type == "U-Net":
                from parameters_UNet import get_params_UNet
                params = get_params_UNet(self.HPC)
            else:
                raise NotImplementedError("Allowed options are: 'CNN', 'CNN_pre', 'U-Net'.")

        # Define an attribute for each parameter
        for k, v in params.items():
            setattr(self, k, v)

    def build_input_pipeline(self, TASK, DEBUG=False, models_test=None, test_folder=None):
        """
        Build the input pipeline
        :param TASK: "TRAIN" or "TEST"
        :param DEBUG: if True: activate verbose output
        :param models_test: if TASK == "TEST": a different set of models (templates) from those for training can be
                            provided here (currently only implemented for CNN)
        :param test_folder: if TASK == "TEST": a different folder to load testing data from can be provided here
                            (currently only implemented for CNN)
        """

        # Define a TF graph
        graph = tf.Graph()

        # Set test folder and models (CNN only)
        MODELS = self["models"]
        if "CNN" in self.NN_type == "CNN":
            test_folder = test_folder or self["data_folder"]
            models_test = models_test or MODELS
        else:
            test_folder = self["data_folder"]
            models_test = MODELS
        
        # First, build the generator for the training data
        PRE_GEN = False

        if self.NN_type == "CNN":
            self.generator = PairGeneratorCNNOnTheFly(template_folder=self["template_path"],
                                                      models=self["models"],
                                                      model_vars=self["model_vars"],
                                                      prior_dict=self["prior_dict"],
                                                      log_priors=self["log_priors"],
                                                      nside=self["nside"],
                                                      indices=self["indexes"],
                                                      mask=self["mask"],
                                                      const_exp=self["const_exp"],
                                                      remove_exp=self["remove_exp"],
                                                      project_2D=self["input_shape"] if self["project_2d"] else None,
                                                      ROI_ring=self["outer_rad"] + self["rad_safety"],
                                                      map_cond=self["cond_on_training_data"],
                                                      flux_cond=self["cond_on_training_labels"],
                                                      n_phot_cond=self["cond_on_training_n_phot"],
                                                      p_cond=self["prob_for_conditions"],
                                                      folder_maps_PS=self["folder_maps_PS"],
                                                      test_fraction_PS=self["test_fraction_PS"],
                                                      combine_gce=self["combine_gce"],
                                                      gce_PS_var=self["gce_PS_var"],
                                                      gce_return_hist=self["gce_return_hist"],
                                                      gce_hist_bins=self["gce_hist_bins"],
                                                      which_hist=self["which_histogram"],
                                                      power_of_F=self["power_of_F"],
                                                      no_PSF=self["no_PSF"],
                                                      test=False)
        elif self.NN_type == "CNN_pre":
            PRE_GEN = True
            self.generator = PairGeneratorCNNPreGenerated(data_folder=self["data_folder"],
                                                          models=self["models"],
                                                          nside=self["nsides"][0],
                                                          ring2nest=self["ring2nest"],
                                                          unmask=self["unmask"],
                                                          indices=None if self["unmask"] else self["indexes"],
                                                          remove_exp=True,  # always remove exposure and work with flux
                                                          project_2D=self["input_shape"] if self["project_2d"] else None,
                                                          ROI_ring=self["outer_rad"] + self["rad_safety"],
                                                          map_cond=self["cond_on_training_data"],
                                                          flux_cond=self["cond_on_training_labels"],
                                                          p_cond=self["prob_for_conditions"],
                                                          test=False)
            if not self["unmask"]:
                print("With this choice of ROI for the neural network,",
                      len(np.setdiff1d(self.generator.settings_dict["unmasked_pix"], self["indexes"][0])),
                      "pixels with data lie outside the ROI.")
                
        elif self.NN_type == "U-Net":
            self.generator = PairGeneratorUNet(template_folder=self["template_path"],
                                               models=self["models"],
                                               model_vars=self["model_vars"],
                                               estimate_templates=self["estimate_templates"],
                                               prior_dict=self["prior_dict"],
                                               log_priors=self["log_priors"],
                                               nside=self["nside"],
                                               indices=self["indexes"],
                                               mask=self["mask"],
                                               const_exp=self["const_exp"],
                                               folder_maps_PS=self["folder_maps_PS"],
                                               test_fraction_PS=self["test_fraction_PS"],
                                               test=False)
        else:
            raise NotImplementedError

        # DEBUG: explicitly construct the iterator
        if DEBUG:
            iter_db = self.generator.get_next_pair(extra_info=True)
            gen_db = next(iter_db)
            print("Shapes: Data:", gen_db["data"].shape, "Labels:", gen_db["label"].shape)
            if not PRE_GEN:
                print("Total number of template variants:", len(gen_db["var_fracs"]))
                if self["const_exp"]:
                    assert np.all(np.abs(np.round(gen_db["data"]) - gen_db["data"]) < 1e-8), \
                        "Not all of the counts are integers although constant exposure is chosen! " \
                        "Something is wrong... aborting!"
            if "CNN" in self.NN_type:
                if self["cond_on_training_labels"] is not None or self["cond_on_training_data"] is not None:
                    n_cond_test = 100
                    maps_db, labels_db = np.zeros((n_cond_test, *gen_db["data"].shape)), np.zeros(
                        (n_cond_test, len(self["models"])))
                    for i_db in range(labels_db.shape[0]):
                        next_db = next(iter_db)
                        maps_db[i_db], labels_db[i_db] = next_db["data"], next_db["label"]
                    if self["cond_on_training_labels"] is not None:
                        cond_true_l = np.asarray(
                            [self["cond_on_training_labels"](labels_db[i]) for i in range(n_cond_test)])
                        print("Condition on labels satisfied for {0}/{1} samples ({2:.2f}).".format(cond_true_l.sum(),
                                                                                                    n_cond_test,
                                                                                                    cond_true_l.sum() / n_cond_test))
                    if self["cond_on_training_data"] is not None:
                        cond_true_m = np.asarray(
                            [self["cond_on_training_data"](maps_db[i]) for i in range(n_cond_test)])
                        print("Condition on maps satisfied for {0}/{1} samples ({2:.2f}).".format(cond_true_m.sum(),
                                                                                                  n_cond_test,
                                                                                                  cond_true_m.sum() / n_cond_test))

        # Define shape of input data
        if self.NN_type == "U-Net":
            data_shape = len(self["indexes"][0])
        else:
            if self["project_2d"]:
                data_shape = self["input_shape"]
            elif self["unmask"]:
                data_shape = None
            else:
                data_shape = len(self["indexes"][0])

        # Build tensorflow dataset from training data
        if PRE_GEN:
            n_variants = 1
        else:
            n_variants = len(flatten_var_fracs(self["model_vars"]))

        if self.NN_type == "U-Net":
            self.ds = DatasetUNet(self.generator, n_params=self["n_params"], n_variants=n_variants,
                                  batch_size=self["batch_size"], prefetch_batch_buffer=self["prefetch_batch_buffer"],
                                  graph=graph, nside=self["nsides"][0], trafo=self["data_trafo"], data_shape=data_shape)
        else:
            params_out = self["n_params"]
            self.ds = DatasetCNN(self.generator, n_params=params_out, n_variants=n_variants,
                                 batch_size=self["batch_size"], prefetch_batch_buffer=self["prefetch_batch_buffer"],
                                 graph=graph, nside=self["nsides"][0], trafo=self["data_trafo"], data_shape=data_shape,
                                 var_fracs=not PRE_GEN, gce_hist=self["gce_return_hist"] and not PRE_GEN,
                                 n_bins_hist=len(self["gce_hist_bins"]) - 1)
        self.input_train = self.ds.next_element
        self.input_train_dict = self.input_train.vars()

        # DEBUG: plot some samples from the tensorflow dataset to check correctness of input pipeline
        # NOTE: Everything should be in NESTED format at this stage.
        # If CNN: the exposure correction might be REMOVED (-> counts are not necessarily integers anymore)
        if DEBUG:
            db_sample = 0
            sess_db = tf.compat.v1.InteractiveSession(graph=graph)

            if "CNN" in self.NN_type:
                debug_out = dict(zip(self.input_train_dict.keys(), sess_db.run(list(self.input_train_dict.values()))))
                debug_data, debug_label = debug_out["data"], debug_out["label"]
                if self["project_2d"]:
                    plt.imshow(debug_data[db_sample, :], origin="lower")
                else:
                    if self["unmask"]:
                        hp.mollview(debug_data[db_sample, :], nest=True)
                    else:
                        hp.mollview(masked_to_full(debug_data[db_sample, :], self["indexes"][0], fill_value=0.0,
                                                   nside=self["nsides"][0]), nest=True)
                ax = plt.gca()
                ax.set_title(
                    [MODELS[i] + ": " + str(np.round(debug_label[db_sample, i], 2)) for i in range(len(MODELS))])

            elif self.NN_type == "U-Net":
                debug_out = dict(zip(self.input_train_dict.keys(), sess_db.run(list(self.input_train_dict.values()))))
                debug_data, debug_label = debug_out["data"], debug_out["label"]
                fig_db, axs_db = plt.subplots(1, len(MODELS) + 2, figsize=(12, 4.5))
                hp.mollview(masked_to_full(debug_data[db_sample, :], self["indexes"][0], fill_value=0.0,
                                           nside=self["nsides"][0]),
                            nest=True, sub=(1, len(MODELS) + 2, 1), title="Total")
                for i_temp in range(len(MODELS)):
                    hp.mollview(masked_to_full(debug_label[db_sample, :, i_temp], self["indexes"][0], fill_value=0.0,
                                               nside=self["nsides"][0]),
                                nest=True, sub=(1, len(MODELS) + 2, 2 + i_temp), notext=True,
                                title=self["model_names"][i_temp])
                hp.mollview(masked_to_full(np.ones_like(debug_data)[0] * (debug_label[db_sample].sum(-1) > 0),
                                           self["indexes"][0],
                                           fill_value=0.0, nside=self["nsides"][0]), nest=True,
                            sub=(1, len(MODELS) + 2, len(MODELS) + 2), title="Relevant pixels")
                [axs_db[i].axis("off") for i in range(len(MODELS) + 2)]

        assert TASK == "TEST" or test_folder == self["data_folder"] and models_test == MODELS, \
            "Choosing a different data folder or models for the test data is only allowed at testing time"

        # Get generator and tensorflow dataset for the test data. Store settings dictionary from training data.
        if "CNN" in self.NN_type:
            if PRE_GEN:
                self.generator_test = PairGeneratorCNNPreGenerated(data_folder=self["data_folder"], models=MODELS,
                                                                   nside=self["nsides"][0], ring2nest=self["ring2nest"],
                                                                   unmask=self["unmask"],
                                                                   indices=None if self["unmask"] else self["indexes"],
                                                                   remove_exp=True,
                                                                   project_2D=self["input_shape"] if self["project_2d"]
                                                                                                  else None,
                                                                   ROI_ring=self["outer_rad"] + self["rad_safety"],
                                                                   test=True)
                self.generator_test.store_settings_dict(self.generator.settings_dict)
            else:
                self.generator_test = PairGeneratorCNNOnTheFly(template_folder=self["template_path"],
                                                               models=models_test,
                                                               model_vars=self["model_vars"],
                                                               prior_dict=self["prior_dict"],
                                                               log_priors=self["log_priors"],
                                                               nside=self["nside"], indices=self["indexes"],
                                                               mask=self["mask"],
                                                               remove_exp=self["remove_exp"],
                                                               const_exp=self["const_exp"],
                                                               project_2D=self["input_shape"] if self[
                                                                   "project_2d"] else None,
                                                               ROI_ring=self["outer_rad"] + self["rad_safety"],
                                                               map_cond=self["cond_on_training_data"],
                                                               flux_cond=self["cond_on_training_labels"],
                                                               n_phot_cond=self["cond_on_training_n_phot"],
                                                               p_cond=self["prob_for_conditions"],
                                                               folder_maps_PS=self["folder_maps_PS"],
                                                               combine_gce=self["combine_gce"],
                                                               gce_PS_var=self["gce_PS_var"],
                                                               gce_return_hist=self["gce_return_hist"],
                                                               gce_hist_bins=self["gce_hist_bins"],
                                                               which_hist=self["which_histogram"],
                                                               power_of_F=self["power_of_F"],
                                                               no_PSF=self["no_PSF"],
                                                               test=True)
                self.generator_test.copy_generator_attributes(self.generator.test_files_PS,
                                                              self.generator.unmasked_pix_PS,
                                                              self.generator.nside_PS,
                                                              self.generator.pixel_mapping_PS_set,
                                                              self.generator.pixel_mapping_PS_get)

            self.ds_test = DatasetCNN(self.generator_test, n_params=self["n_params"], n_variants=n_variants,
                                      batch_size=self["batch_size_val"],
                                      prefetch_batch_buffer=self["prefetch_batch_buffer_val"], graph=graph,
                                      nside=self["nsides"][0],
                                      trafo=self["data_trafo"], data_shape=data_shape, var_fracs=not PRE_GEN,
                                      gce_hist=self["gce_return_hist"] and not PRE_GEN,
                                      n_bins_hist=len(self["gce_hist_bins"]) - 1)

        elif self.NN_type == "U-Net":
            self.generator_test = PairGeneratorUNet(template_folder=self["template_path"], models=models_test,
                                                    model_vars=self["model_vars"],
                                                    estimate_templates=self["estimate_templates"],
                                                    prior_dict=self["prior_dict"], log_priors=self["log_priors"],
                                                    nside=self["nside"], indices=self["indexes"], mask=self["mask"],
                                                    const_exp=self["const_exp"], folder_maps_PS=self["folder_maps_PS"],
                                                    test_fraction_PS=self["test_fraction_PS"], test=True)
            self.generator_test.copy_generator_attributes(self.generator.test_files_PS, self.generator.unmasked_pix_PS,
                                                          self.generator.nside_PS, self.generator.pixel_mapping_PS_set,
                                                          self.generator.pixel_mapping_PS_get)
            self.ds_test = DatasetUNet(self.generator_test, n_params=self["n_params"], n_variants=n_variants,
                                       batch_size=self["batch_size_val"],
                                       prefetch_batch_buffer=self["prefetch_batch_buffer_val"], graph=graph,
                                       nside=self["nsides"][0], trafo=self["data_trafo"], data_shape=data_shape)

        self.input_test = self.ds_test.next_element
        self.input_test_db = self.ds_test.next_element_with_info

        # Get Fermi count data for analysis after training
        if "CNN" in self.NN_type and not self["project_2d"]:
            if PRE_GEN:
                rescale_fermi_counts = self.generator_test.settings_dict["rescale"]
                nside_fermi_counts = 128
            else:
                rescale_fermi_counts = self.generator_test.fermi_rescale_to_apply
                nside_fermi_counts = self["nside"]
            self.fermi_counts = self.ds_test.get_fermi_counts(self["template_path"], fermi_name="fermidata_counts",
                                                    rescale=rescale_fermi_counts, indices=self["indexes"][0],
                                                    outer_mask=self["ROI_fermi"], mask_3FGL=self["mask_3FGL_fermi"],
                                                    nside=nside_fermi_counts, remove_exp=self["remove_exp"])

    def build_model(self):
        """
        Build the neural network
        """
        if self.NN_type == "U-Net":
            self.model = models.graphUNet(**self.__dict__)
        elif "CNN" in self.NN_type:
            if self["project_2d"] == 2:  # EfficientNet
                self.model = models.EfficientNetWrapper(**self.__dict__)
            elif self["project_2d"] == 1:  # Standard CNN
                self.model = models.cnn2d(**self.__dict__)
            else:  # Deepsphere
                self.model = models.deepsphere(**self.__dict__)

        # Print NN parameters
        print("NN parameters:")
        pprint(self.__dict__)

    def train(self, RESUME=False):
        """
        Train neural network
        :param RESUME: resume NN training?
        """
        assert RESUME or (self["alea_step"] != 2 and not self["use_ffjord"]), \
            "Set RESUME = True when fitting the aleatoric uncertainty covariance or FFJORD!"

        if not RESUME:
            # Cleanup before running again.
            shutil.rmtree(self.model.get_path("summaries"), ignore_errors=True)
            shutil.rmtree(self.model.get_path("checkpoints"), ignore_errors=True)

        # Backup source files
        mkdir_p(self.model.get_path("checkpoints"))
        backup_train_files(__file__, self.model.get_path("checkpoints"), self.NN_type)

        # Save params dictionary
        datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
        params_file = open(os.path.join(self.model.get_path("checkpoints"), "params_" + datetime + ".pickle"), "wb")
        try:
            params_file.write(dill.dumps(self.params_orig))
        except TypeError:
            params_file.write(cloudpickle.dumps(self.params_orig))
        params_file.close()

        # Train
        print("\n\n===== STARTING NN TRAINING =====")
        # Train only last layer?
        only_last_layer = False
        # Option 1: first means only, then full covariance matrix (Gaussian)
        if self["aleatoric"] and self["alea_split"]:
            only_last_layer = True
        # Option 2: first means / means + diag. Gaussian uncertainties, then FFJORD
        if self["build_ffjord"] and self["use_ffjord"]:
            only_last_layer = True

        accuracy_validation, loss_validation, loss_training, t_step = self.model.fit(resume=RESUME,
                                                                                     log_device_placement=self.HPC,
                                                                                     only_last_layer=only_last_layer)

        # Plot loss
        try:
            logy = not self["aleatoric"] or (self["alea_split"] and self["alea_step"] == 1)
            plot.plot_loss(loss_training, loss_validation, t_step, self['eval_frequency'], logy=logy)
            fig = plt.gcf()
            fig.savefig(os.path.join(self.model.get_path("checkpoints"), "loss.pdf"))
        except:
            print("WARNING: loss plot could not be generated!")

    def predict_cnn(self, sess, data, epi_n_dropout_analysis=1):
        # Initialise (not everything might be set)
        pred_fluxes_dict, pred_fluxes, covar_fluxes, covar_fluxes_aleat, covar_fluxes_epist, var_fracs \
                   = {}, None, None, None, None, None

        # Get predictions for test samples
        if self["aleatoric"] and not self.model.gaussian_mixture:
            if self["epistemic"]:
                pred_fluxes_dict = [self.model.predict(data, sess) for _ in range(epi_n_dropout_analysis)]
                pred_fluxes_dict = combine_list_of_dicts(pred_fluxes_dict)
                pred_fluxes_mc, covar_fluxes_mc = pred_fluxes_dict["logits_mean"], pred_fluxes_dict["covar"]
                if self.model.estimate_var_fracs:
                    var_fracs_mc = pred_fluxes_dict["var_fracs"]

                # Calculate means and variances
                pred_fluxes = pred_fluxes_mc.mean(0)
                covar_fluxes_aleat = covar_fluxes_mc.mean(0)
                if self.model.estimate_var_fracs:
                    var_fracs = var_fracs_mc.mean(0)

                # if diagonal matrix is assumed for aleatoric uncertainty: also assume diagonal matrix for epistemic uncertainty!
                if self["alea_only_var"]:
                    covar_fluxes_epist = np.asarray(
                        [np.diag(pred_fluxes_mc.var(0)[i, :]) for i in range(n_samples_analysis)])
                else:
                    epist_term_1 = np.mean((np.expand_dims(pred_fluxes_mc, -1) @ np.transpose(
                        np.expand_dims(pred_fluxes_mc, -1), [0, 1, 3, 2])), 0)
                    epist_term_2 = (np.mean((np.expand_dims(pred_fluxes_mc, -1)), 0)) @ (
                        np.mean(np.transpose(np.expand_dims(pred_fluxes_mc, -1), [0, 1, 3, 2]), 0))
                    covar_fluxes_epist = epist_term_1 - epist_term_2

                covar_fluxes = covar_fluxes_aleat + covar_fluxes_epist
            else:
                pred_fluxes_dict = self.model.predict(data, sess)
                pred_fluxes, covar_fluxes = pred_fluxes_dict["logits_mean"], pred_fluxes_dict["covar"]
                if self.model.estimate_var_fracs:
                    var_fracs = pred_fluxes_dict["var_fracs"]
        else:
            pred_fluxes_dict = self.model.predict(data, sess)
            pred_fluxes = pred_fluxes_dict["logits_mean"]
            if self.model.estimate_var_fracs:
                var_fracs = pred_fluxes_dict["var_fracs"]

        return pred_fluxes_dict, pred_fluxes, covar_fluxes, covar_fluxes_aleat, covar_fluxes_epist, var_fracs

    def quick_evaluate(self, models_test=None, model_names=None, filter_FF_min=0.3, n_samples_analysis=128,
                       epi_n_dropout_analysis=10, test_checkpoint=None, save_figs=True):
        if models_test is None:
            models_test = self["models"]
        if model_names is None:
            model_names = self["model_names"]
        sess_test = None if test_checkpoint is None else self.model.get_session(checkpoint=test_checkpoint)
        input_test_dict = self.ds_test.next_element.vars()

        # U-Net
        if self.NN_type == "U-Net":
            UM = self.generator_test.unmask_fct
            test_data, test_label, test_var_fracs = self.ds_test.get_samples(n_samples_analysis)

            # Predict
            pred_fluxes_dict, pred_loss = self.model.predict(test_data, test_label, test_var_fracs, sess_test)
            pred_frac = pred_fluxes_dict["logits_mean"]
            if self.model.estimate_var_fracs:
                var_fracs = pred_fluxes_dict["var_fracs"]
            pred_counts = pred_frac * np.expand_dims(test_data, -1)

            # Plot
            n_to_plot = 1
            unet_evaluate(test_label, pred_counts, models_test, model_names, UM, marg=0.02, plot_inds=range(n_to_plot),
                          folder=self.model.get_path("checkpoints"))

        # CNN
        elif "CNN" in self.NN_type:
            # Pre-generated data: variants are not supported
            test_out = self.ds_test.get_samples(n_samples_analysis)
            test_data, test_label = test_out["data"], test_out["label"]
            if "var_fracs" in test_out.keys():
                test_var_fracs = test_out["var_fracs"]
            if "gce_hist" in test_out.keys():
                test_gce_hist = test_out["gce_hist"]  # TODO: make a plot
            real_fluxes = test_label

            # Get predictions
            pred_fluxes_dict, pred_fluxes, covar_fluxes, covar_fluxes_aleat, covar_fluxes_epist, var_fracs = \
                        self.predict_cnn(sess_test, test_out, epi_n_dropout_analysis=epi_n_dropout_analysis)

            # Make error plot
            covar_fluxes = None if not self["aleatoric"] or self.model.gaussian_mixture else covar_fluxes
            ms = None if not self["aleatoric"] else 10
            out_file = os.path.join(self.model.get_path("checkpoints"), "error_plot.pdf") if save_figs else None
            make_error_plot(models_test, real_fluxes, pred_fluxes, model_names=model_names, out_file=out_file,
                                                 show_stats=True, colours=None, legend=True, pred_covar=covar_fluxes, ms=ms)

            # Make a plot of the Gaussian mixtures
            if self.model.gaussian_mixture:
                n_plot = 5
                plot_gaussian_mixtures(np.exp(pred_fluxes_dict["logalpha_mixture"])[:n_plot], pred_fluxes_dict["logits_mean_mixture"][:n_plot],
                                       np.sqrt(np.exp(pred_fluxes_dict["logvar_mixture"]))[:n_plot], truths=test_label[:n_plot],
                                       truncated=self["truncate_gaussians"])

            # Make error plot for the template fractions
            if self.model.estimate_var_fracs:
                test_var_fracs_relevant, models_relevant, ind_relevant = get_relevant_var_fracs(test_var_fracs, self.model.model_vars, models_test)
                pred_var_fracs_relevant, _, _ = get_relevant_var_fracs(var_fracs, self.model.model_vars, models_test)
                for i_temp in range(len(ind_relevant)):
                    ind_rel_temp = ind_relevant[i_temp]
                    colour_var_fracs = test_label[:, np.repeat(ind_relevant[i_temp], len(self.model.model_vars[ind_relevant[i_temp]]))]
                    ind2plot = np.argwhere(test_label[:, ind_rel_temp] >= filter_FF_min).flatten()
                    out_file = os.path.join(self.model.get_path("checkpoints"), "var_frac_" + models_test[i_temp] + ".pdf") if save_figs else None
                    make_error_plot(np.asarray(models_relevant)[ind_rel_temp], test_var_fracs_relevant[ind_rel_temp][ind2plot],
                                    pred_var_fracs_relevant[ind_rel_temp][ind2plot], model_names=models_relevant[ind_rel_temp], out_file=out_file,
                                    show_stats=True, colours=(colour_var_fracs[ind2plot, :]).T, legend=True, pred_covar=None, ms=None, alpha=1)


