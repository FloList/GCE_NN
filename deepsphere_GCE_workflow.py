"""
This file contains high-level function that provides NN functionality such as building and training the NN.
"""
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_data import PairGeneratorCNNPreGenerated, PairGeneratorCNNOnTheFly, PairGeneratorUNet, DatasetCNN, DatasetUNet
from gce_utils import *
from deepsphere import models
import os
import dill
import cloudpickle
import seaborn as sns
from pprint import pprint
sns.set_style("white")
sns.set_context("talk")
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'rocket'


################################
# BUILD NN
def build_NN(NN_TYPE, HPC, DEBUG, TASK, TEST_EXP_PATH, test_folder=None, models_test=None, PRE_GEN=False,
             parameter_filename=None, no_hist_for_testing=False):
    # ########################################################
    if HPC:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if HPC == 1:
            sys.path.append("/scratch/u95/fl9575/GCE_v2")
        elif HPC == 2:
            sys.path.append("/project/dss/GCE_NN")
    else:
        pass  # sys.path.append("/home/flo/PycharmProjects/GCE/DeepSphere"), only needed if not running in PyCharm
    # ########################################################

    # Get parameters for the respective NN
    # if training or no NN path is provided: take parameters from parameters_*.py
    if TASK == "TRAIN" or TEST_EXP_PATH is None:
        # Get default names of parameter file and function
        parameter_filename_def, par_fun_name = get_default_param_file_and_fun(NN_TYPE, PRE_GEN)

        # Get name of the parameter file: if provided as a function argument: priority!
        # otherwise, try to see if second argument has been provided
        if parameter_filename is None and len(sys.argv) > 1:  # if parameter_file
            parameter_filename = sys.argv[1]

        if parameter_filename is None:
            parameter_filename = parameter_filename_def

        par_fun = import_from(parameter_filename, par_fun_name)
        params = par_fun(HPC)
        print("Parameters loaded from", "'"+parameter_filename+"'.")

        # Store name of parameter file
        params["param_filename"] = parameter_filename

    # if testing and a NN name is provided: load the parameters
    else:
        params = load_params_from_pickle(TEST_EXP_PATH)
        print("  Loaded NN parameters from", "'"+TEST_EXP_PATH+"'.")
        assert NN_TYPE == params["NN_type"], "It seems like something is wrong with the NN type... aborting!"

    MODELS = params["models"]

    # Set test folder and models (CNN only)
    if NN_TYPE == "CNN":
        if PRE_GEN:
            test_folder = test_folder or params["data_folder"]
        else:
            test_folder = test_folder or params["template_path"]
        models_test = models_test or MODELS
    else:
        test_folder = params["data_folder"]
        models_test = MODELS

    # Define a TF graph
    graph = tf.Graph()

    # Build generator for training data
    if NN_TYPE == "CNN":
        if params["loss"] == "LLH_RATIO":
            pp = params["pp"]
        else:
            pp = None

        if PRE_GEN:
            generator = PairGeneratorCNNPreGenerated(data_folder=params["data_folder"], models=MODELS, nside=params["nside"],
                                                     ring2nest=params["ring2nest"], unmask=params["unmask"],
                                                     indices=None if params["unmask"] else params["indexes"], remove_exp=params["remove_exp"],
                                                     project_2D=params["input_shape"] if params["project_2d"] else None,
                                                     ROI_ring=params["outer_rad"] + params["rad_safety"],
                                                     map_cond=params["cond_on_training_data"], flux_cond=params["cond_on_training_labels"],
                                                     p_cond=params["prob_for_conditions"], gce_return_hist=params["gce_return_hist"],
                                                     which_hist=params["which_histogram"], hist_templates=params["gce_hist_templates"],
                                                     pp=pp, test=False)
            if not params["unmask"]:
                print("With this choice of ROI for the neural network,",
                      len(np.setdiff1d(generator.settings_dict["unmasked_pix"], params["indexes"][0])),
                      "pixels with data lie outside the ROI.")

        else:
            generator = PairGeneratorCNNOnTheFly(template_folder=params["template_path"], models=MODELS, model_vars=params["model_vars"],
                                                 prior_dict=params["prior_dict"], log_priors=params["log_priors"], nside=params["nside"],
                                                 indices=params["indexes"], mask=params["mask"], const_exp=params["const_exp"],
                                                 remove_exp=params["remove_exp"], project_2D=params["input_shape"] if params["project_2d"] else None,
                                                 ROI_ring=params["outer_rad"] + params["rad_safety"], map_cond=params["cond_on_training_data"],
                                                 flux_cond=params["cond_on_training_labels"], n_phot_cond=params["cond_on_training_n_phot"],
                                                 p_cond=params["prob_for_conditions"], folder_maps_PS=params["folder_maps_PS"],
                                                 test_fraction_PS=params["test_fraction_PS"], combine_gce=params["combine_gce"],
                                                 gce_PS_var=params["gce_PS_var"], gce_return_hist=params["gce_return_hist"],
                                                 gce_hist_bins=params["gce_hist_bins"], which_hist=params["which_histogram"],
                                                 power_of_F=params["power_of_F"], no_PSF=params["no_PSF"],
                                                 pp=pp, test=False)

    elif NN_TYPE == "U-Net":
        generator = PairGeneratorUNet(template_folder=params["template_path"], models=MODELS, model_vars=params["model_vars"],
                                      estimate_templates=params["estimate_templates"], prior_dict=params["prior_dict"],
                                      log_priors=params["log_priors"], nside=params["nside"], indices=params["indexes"],
                                      mask=params["mask"], const_exp=params["const_exp"], folder_maps_PS=params["folder_maps_PS"],
                                      test_fraction_PS=params["test_fraction_PS"], test=False)
    else:
        raise NotImplementedError("This NN type is not implemented!")

    # DEBUG: explicitly construct the iterator
    if DEBUG:
        iter_db = generator.get_next_pair(extra_info=True)
        gen_db = next(iter_db)
        print("Shapes: Data:", gen_db["data"].shape, "Labels:", gen_db["label"].shape)
        if not PRE_GEN:
            print("Total number of template variants:", len(gen_db["var_fracs"]))
            if params["const_exp"]:
                assert np.all(np.abs(np.round(gen_db["data"]) - gen_db["data"]) < 1e-8), \
                    "Not all of the counts are integers although constant exposure is chosen! Something is wrong... aborting!"
        if NN_TYPE == "CNN":
            if params["cond_on_training_labels"] is not None or params["cond_on_training_data"] is not None:
                n_cond_test = 100
                maps_db, labels_db = np.zeros((n_cond_test, *gen_db["data"].shape)), np.zeros((n_cond_test, len(MODELS)))
                for i_db in range(labels_db.shape[0]):
                    next_db = next(iter_db)
                    maps_db[i_db], labels_db[i_db] = next_db["data"], next_db["label"]
                if params["cond_on_training_labels"] is not None:
                    cond_true_l = np.asarray([params["cond_on_training_labels"](labels_db[i]) for i in range(n_cond_test)])
                    print("Condition on labels satisfied for {0}/{1} samples ({2:.2f}).".format(cond_true_l.sum(), n_cond_test, cond_true_l.sum() / n_cond_test))
                if params["cond_on_training_data"] is not None:
                    cond_true_m = np.asarray([params["cond_on_training_data"](maps_db[i]) for i in range(n_cond_test)])
                    print("Condition on maps satisfied for {0}/{1} samples ({2:.2f}).".format(cond_true_m.sum(), n_cond_test, cond_true_m.sum() / n_cond_test))

    # Define shape of input data
    if NN_TYPE == "U-Net":
        data_shape = len(params["indexes"][0])
    else:
        if params["project_2d"]:
            data_shape = params["input_shape"]
        elif params["unmask"]:
            data_shape = None
        else:
            data_shape = len(params["indexes"][0])

    # Build tensorflow dataset from training data
    if PRE_GEN:
        n_variants = 1
    else:
        n_variants = len(flatten_var_fracs(params["model_vars"]))

    # Number of parameters:
    if NN_TYPE == "CNN" and params["loss"] == "LLH_RATIO":
        n_params = params["pp"].get_n_params()
    else:
        n_params = params["n_params"]

    if NN_TYPE == "U-Net":
        ds = DatasetUNet(generator, n_params=n_params, n_variants=n_variants, batch_size=params["batch_size"],
                         prefetch_batch_buffer=params["prefetch_batch_buffer"], graph=graph, nside=params["nsides"][0],
                         trafo=params["data_trafo"], data_shape=data_shape)
    else:
        # Set no. of histogram bins
        if params["gce_return_hist"]:
            if PRE_GEN:
                params["gce_hist_bins"] = generator.gce_hist_bins
                n_channels_hist = len(params["gce_hist_templates"])
            else:
                n_channels_hist = 1  # only 1 histogram is supported for on-the-fly data generation
            n_bins_hist = len(params["gce_hist_bins"]) - 1
        else:
            n_bins_hist = None
            n_channels_hist = 1
        ds = DatasetCNN(generator, n_params=n_params, n_variants=n_variants, batch_size=params["batch_size"],
                        prefetch_batch_buffer=params["prefetch_batch_buffer"], graph=graph, nside=params["nside"],
                        trafo=params["data_trafo"], data_shape=data_shape, var_fracs=not PRE_GEN,
                        gce_hist=params["gce_return_hist"], n_bins_hist=n_bins_hist, n_channels_hist=n_channels_hist)
    input_train = ds.next_element
    input_train_dict = input_train.vars()

    # DEBUG: plot some samples from the tensorflow dataset to check correctness of input pipeline
    # NOTE: Everything should be in NESTED format at this stage.
    # If CNN: the exposure correction could be REMOVED (-> counts are no integers anymore)
    if DEBUG:
        db_sample = 0
        sess_db = tf.compat.v1.InteractiveSession(graph=graph)

        if NN_TYPE == "CNN":
            debug_out = dict(zip(input_train_dict.keys(), sess_db.run(list(input_train_dict.values()))))
            debug_data, debug_label = debug_out["data"], debug_out["label"]
            if params["project_2d"]:
                plt.imshow(debug_data[db_sample, :], origin="lower")
            else:
                if params["unmask"]:
                    hp.mollview(debug_data[db_sample, :], nest=True)
                else:
                    hp.mollview(masked_to_full(debug_data[db_sample, :], params["indexes"][0], fill_value=0.0,
                                               nside=params["nsides"][0]), nest=True)
            ax = plt.gca()
            ax.set_title([MODELS[i] + ": " + str(np.round(debug_label[db_sample, i], 2)) for i in range(len(MODELS))])

        elif NN_TYPE == "U-Net":
            debug_out = dict(zip(input_train_dict.keys(), sess_db.run(list(input_train_dict.values()))))
            debug_data, debug_label = debug_out["data"], debug_out["label"]
            fig_db, axs_db = plt.subplots(1, len(MODELS) + 2, figsize=(12, 4.5))
            hp.mollview(masked_to_full(debug_data[db_sample, :], params["indexes"][0], fill_value=0.0, nside=params["nsides"][0]),
                        nest=True, sub=(1, len(MODELS) + 2, 1), title="Total")
            for i_temp in range(len(MODELS)):
                hp.mollview(masked_to_full(debug_label[db_sample, :, i_temp], params["indexes"][0], fill_value=0.0, nside=params["nsides"][0]),
                            nest=True, sub=(1, len(MODELS) + 2, 2 + i_temp), notext=True, title=params["model_names"][i_temp])
            hp.mollview(masked_to_full(np.ones_like(debug_data)[0] * (debug_label[db_sample].sum(-1) > 0), params["indexes"][0],
                                       fill_value=0.0, nside=params["nsides"][0]), nest=True, sub=(1, len(MODELS) + 2, len(MODELS) + 2), title="Relevant pixels")
            [axs_db[i].axis("off") for i in range(len(MODELS) + 2)]

    if NN_TYPE == "CNN" and PRE_GEN:
        assert TASK == "TEST" or test_folder == params["data_folder"] and models_test == MODELS, \
            "Choosing a different data folder or models for the test data is only allowed at testing time"
    elif NN_TYPE == "CNN" and not PRE_GEN:
        assert TASK == "TEST" or test_folder == params["template_path"] and models_test == MODELS, \
            "Choosing a different data folder or models for the test data is only allowed at testing time"

    # Get generator and tensorflow dataset for the test data. Store settings dictionary from training data.
    if NN_TYPE == "CNN":
        return_test_hist = False if no_hist_for_testing else params["gce_return_hist"]
        if PRE_GEN:
            generator_test = PairGeneratorCNNPreGenerated(data_folder=test_folder, models=models_test, nside=params["nside"],
                                                          ring2nest=params["ring2nest"], unmask=params["unmask"],
                                                          indices=None if params["unmask"] else params["indexes"], remove_exp=params["remove_exp"],
                                                          project_2D=params["input_shape"] if params["project_2d"] else None,
                                                          ROI_ring=params["outer_rad"] + params["rad_safety"],
                                                          map_cond=params["cond_on_training_data"], flux_cond=params["cond_on_training_labels"],
                                                          p_cond=params["prob_for_conditions"], gce_return_hist=return_test_hist,
                                                          which_hist=params["which_histogram"], hist_templates=params["gce_hist_templates"],
                                                          pp=pp, test=True)
            generator_test.store_settings_dict(generator.settings_dict)
            generator_test.copy_generator_attributes(generator)
        else:
            generator_test = PairGeneratorCNNOnTheFly(template_folder=test_folder, models=models_test,
                                                      model_vars=params["model_vars"], prior_dict=params["prior_dict"],
                                                      log_priors=params["log_priors"], nside=params["nside"],
                                                      indices=params["indexes"], mask=params["mask"],
                                                      remove_exp=params["remove_exp"], const_exp=params["const_exp"],
                                                      project_2D=params["input_shape"] if params["project_2d"] else None,
                                                      ROI_ring=params["outer_rad"] + params["rad_safety"], map_cond=params["cond_on_training_data"],
                                                      flux_cond=params["cond_on_training_labels"], n_phot_cond=params["cond_on_training_n_phot"],
                                                      p_cond=params["prob_for_conditions"], folder_maps_PS=params["folder_maps_PS"],
                                                      combine_gce=params["combine_gce"], gce_PS_var=params["gce_PS_var"],
                                                      gce_return_hist=params["gce_return_hist"], gce_hist_bins=return_test_hist,
                                                      which_hist=params["which_histogram"], power_of_F=params["power_of_F"],
                                                      no_PSF=params["no_PSF"], pp=pp,
                                                      test=True)
            generator_test.copy_generator_attributes(generator.test_files_PS, generator.unmasked_pix_PS, generator.nside_PS,
                                                     generator.pixel_mapping_PS_set, generator.pixel_mapping_PS_get)

        ds_test = DatasetCNN(generator_test, n_params=n_params, n_variants=n_variants, batch_size=params["batch_size_val"],
                             prefetch_batch_buffer=params["prefetch_batch_buffer_val"], graph=graph, nside=params["nside"],
                             trafo=params["data_trafo"], data_shape=data_shape, var_fracs=not PRE_GEN,
                             gce_hist=return_test_hist, n_bins_hist=n_bins_hist, n_channels_hist=n_channels_hist)

    elif NN_TYPE == "U-Net":
        generator_test = PairGeneratorUNet(template_folder=params["template_path"], models=models_test, model_vars=params["model_vars"],
                                           estimate_templates=params["estimate_templates"], prior_dict=params["prior_dict"], log_priors=params["log_priors"],
                                           nside=params["nside"], indices=params["indexes"], mask=params["mask"], const_exp=params["const_exp"],
                                           folder_maps_PS=params["folder_maps_PS"], test_fraction_PS=params["test_fraction_PS"], test=True)
        generator_test.copy_generator_attributes(generator.test_files_PS, generator.unmasked_pix_PS, generator.nside_PS,
                                                 generator.pixel_mapping_PS_set, generator.pixel_mapping_PS_get)
        ds_test = DatasetUNet(generator_test, n_params=params["n_params"], n_variants=n_variants,
                              batch_size=params["batch_size_val"],
                              prefetch_batch_buffer=params["prefetch_batch_buffer_val"], graph=graph, nside=params["nsides"][0],
                              trafo=params["data_trafo"], data_shape=data_shape)

    input_test = ds_test.next_element
    input_test_db = ds_test.next_element_with_info

    # Get Fermi count data for analysis after training
    if NN_TYPE == "CNN" and not params["project_2d"]:
        if PRE_GEN:
            rescale_fermi_counts = generator_test.settings_dict["rescale"]
            nside_fermi_counts = params["nside"]
        else:
            rescale_fermi_counts = generator_test.fermi_rescale_to_apply
            nside_fermi_counts = params["nside"]
        fermi_counts = ds_test.get_fermi_counts(params["template_path"], fermi_name="fermidata_counts",
                                                rescale=rescale_fermi_counts, indices=params["indexes"][0],
                                                outer_mask=params["ROI_fermi"], mask_type_fermi=params["mask_type_fermi"],
                                                remove_exp=params["remove_exp"])
    else:
        fermi_counts = None
    ################################
    # Build model
    if NN_TYPE == "U-Net":
        model = models.graphUNet(input_train, input_test, **params)
    elif NN_TYPE == "CNN":
        if params["project_2d"] == 1:  # Standard CNN
            model = models.cnn2d(input_train, input_test, **params)
        else:  # Deepsphere
            model = models.deepsphere(input_train, input_test, **params)

        # Store prediction parameter object as a model attributes
        model.pp = pp

    # Print NN parameters
    print("NN parameters:")
    pprint(params)

    # Return model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts
    return model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts


################################
# TRAIN NN
def train_NN(model, params, NN_TYPE, HPC, RESUME=False):
    assert RESUME or (params["alea_step"] != 2), \
        "Set RESUME = True when fitting the aleatoric uncertainty covariance!"
    assert RESUME or (params["gce_hist_step"] != 2), \
        "Set RESUME = True when fitting the histograms after training the FFs only!"

    if not RESUME:
        # Cleanup before running again.
        shutil.rmtree(model.get_path("summaries"), ignore_errors=True)
        shutil.rmtree(model.get_path("checkpoints"), ignore_errors=True)

    # Backup source files
    mkdir_p(model.get_path("checkpoints"))
    param_file = None if "param_filename" not in params.keys() else params["param_filename"] + ".py"
    backup_train_files(__file__, model.get_path("checkpoints"), NN_TYPE, param_file=param_file)

    # Save params dictionary
    datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
    params_file = open(os.path.join(model.get_path("checkpoints"), "params_" + datetime + ".pickle"), "wb")
    try:
        params_file.write(dill.dumps(params))
    except TypeError:
        params_file.write(cloudpickle.dumps(params))
    params_file.close()

    # Train
    print("\n\n===== STARTING NN TRAINING =====")
    # Train only last layer?
    only_last_layer = False
    # first means only, then full covariance matrix (Gaussian)?
    if params["aleatoric"] and params["alea_split"]:
        only_last_layer = True
    # Train only one step for the histogram training?
    if params["gce_return_hist"] and params["gce_hist_split"]:
        only_hist_step = params["gce_hist_step"]
    else:
        only_hist_step = None
    accuracy_validation, loss_validation, loss_training, t_step = model.fit(resume=RESUME, log_device_placement=HPC,
                                                                            only_last_layer=only_last_layer,
                                                                            only_hist_step=only_hist_step)

    # Plot loss
    # if not params["gce_only_hist"]:
    #     logy = not params["aleatoric"] or (params["alea_split"] and params["alea_step"] == 1)
    #     plot.plot_loss(loss_training, loss_validation, t_step, params['eval_frequency'], logy=logy)
    #     fig = plt.gcf()
    #     fig.savefig(os.path.join(model.get_path("checkpoints"), "loss.pdf"))


################################
# Quick evaluation:
def quick_evaluate_NN(NN_TYPE, model, params, generator_test, ds_test, models_test=None, filter_FF_min=0.3,
                      n_samples_analysis=128, epi_n_dropout_analysis=5, TEST_CHECKPOINT=None):
    if models_test is None:
        models_test = params["models"]
    model_names = params["model_names"]
    sess_test = None if TEST_CHECKPOINT is None else model.get_session(checkpoint=TEST_CHECKPOINT)
    input_test_dict = ds_test.next_element.vars()

    # U-Net
    if NN_TYPE == "U-Net":
        UM = generator_test.unmask_fct
        test_data, test_label, test_var_fracs = ds_test.get_samples(n_samples_analysis)

        # Predict
        pred_fluxes_dict, pred_loss = model.predict(test_data, test_label, test_var_fracs, sess_test)
        pred_frac = pred_fluxes_dict["logits_mean"]
        if model.estimate_var_fracs:
            var_fracs = pred_fluxes_dict["var_fracs"]
        pred_counts = pred_frac * np.expand_dims(test_data, -1)

        # Plot
        n_to_plot = 1
        unet_evaluate(test_label, pred_counts, models_test, model_names, UM, marg=0.02, plot_inds=range(n_to_plot),
                      folder=model.get_path("checkpoints"))

    # CNN
    elif NN_TYPE == "CNN":
        # Pre-generated data: variants are not supported
        test_out = ds_test.get_samples(n_samples_analysis)
        test_data, test_label = test_out["data"], test_out["label"]
        if "var_fracs" in test_out.keys():
            test_var_fracs = test_out["var_fracs"]
        if "gce_hist" in test_out.keys():
            test_gce_hist = test_out["gce_hist"]
        real_fluxes = test_label

        # Get predictions for test samples
        if params["aleatoric"] and not model.gaussian_mixture:
            if params["epistemic"]:
                pred_fluxes_dict = [model.predict(test_out, sess_test) for _ in range(epi_n_dropout_analysis)]
                pred_fluxes_dict = combine_list_of_dicts(pred_fluxes_dict)
                pred_fluxes_mc, covar_fluxes_mc = pred_fluxes_dict["logits_mean"], pred_fluxes_dict["covar"]
                if model.estimate_var_fracs:
                    var_fracs_mc = pred_fluxes_dict["var_fracs"]

                # Calculate means and variances
                pred_fluxes = pred_fluxes_mc.mean(0)
                covar_fluxes_aleat = covar_fluxes_mc.mean(0)
                if model.estimate_var_fracs:
                    var_fracs = var_fracs_mc.mean(0)

                # if diagonal matrix is assumed for aleatoric uncertainty: also assume diagonal matrix for epistemic uncertainty!
                if params["alea_only_var"]:
                    covar_fluxes_epist = np.asarray([np.diag(pred_fluxes_mc.var(0)[i, :]) for i in range(n_samples_analysis)])
                else:
                    epist_term_1 = np.mean((np.expand_dims(pred_fluxes_mc, -1) @ np.transpose(np.expand_dims(pred_fluxes_mc, -1), [0, 1, 3, 2])), 0)
                    epist_term_2 = (np.mean((np.expand_dims(pred_fluxes_mc, -1)), 0)) @ (np.mean(np.transpose(np.expand_dims(pred_fluxes_mc, -1), [0, 1, 3, 2]), 0))
                    covar_fluxes_epist = epist_term_1 - epist_term_2

                covar_fluxes = covar_fluxes_aleat + covar_fluxes_epist
            else:
                pred_fluxes_dict = model.predict(test_out, sess_test)
                pred_fluxes, covar_fluxes = pred_fluxes_dict["logits_mean"], pred_fluxes_dict["covar"]
                if model.estimate_var_fracs:
                    var_fracs = pred_fluxes_dict["var_fracs"]
        else:
            pred_fluxes_dict = model.predict(test_out, sess_test)
            pred_fluxes = pred_fluxes_dict["logits_mean"]
            if model.estimate_var_fracs:
                var_fracs = pred_fluxes_dict["var_fracs"]

        # Make error plot
        covar_fluxes = None if not params["aleatoric"] or model.gaussian_mixture else covar_fluxes
        ms = None if not params["aleatoric"] else 10
        out_file = os.path.join(model.get_path("checkpoints"), "error_plot.pdf")
        make_error_plot(models_test, real_fluxes, pred_fluxes, model_names=model_names, out_file=out_file,
                        show_stats=True, colours=None, legend=True, pred_covar=covar_fluxes, ms=ms)

        # Make a plot of the Gaussian mixtures
        if model.gaussian_mixture:
            n_plot = 5
            plot_gaussian_mixtures(np.exp(pred_fluxes_dict["logalpha_mixture"])[:n_plot], pred_fluxes_dict["logits_mean_mixture"][:n_plot],
                                   np.sqrt(np.exp(pred_fluxes_dict["logvar_mixture"]))[:n_plot], truths=test_label[:n_plot],
                                   truncated=params["truncate_gaussians"])

        # Make error plot for the template fractions
        if model.estimate_var_fracs:
            test_var_fracs_relevant, models_relevant, ind_relevant = get_relevant_var_fracs(test_var_fracs, model.model_vars, models_test)
            pred_var_fracs_relevant, _, _ = get_relevant_var_fracs(var_fracs, model.model_vars, models_test)
            for i_temp in range(len(ind_relevant)):
                ind_rel_temp = ind_relevant[i_temp]
                colour_var_fracs = test_label[:, np.repeat(ind_relevant[i_temp], len(model.model_vars[ind_relevant[i_temp]]))]
                ind2plot = np.argwhere(test_label[:, ind_rel_temp] >= filter_FF_min).flatten()
                out_file = os.path.join(model.get_path("checkpoints"), "var_frac_" + models_test[i_temp] + ".pdf")
                make_error_plot(np.asarray(models_relevant)[ind_rel_temp], test_var_fracs_relevant[ind_rel_temp][ind2plot],
                                pred_var_fracs_relevant[ind_rel_temp][ind2plot], model_names=models_relevant[ind_rel_temp], out_file=out_file,
                                show_stats=True, colours=(colour_var_fracs[ind2plot, :]).T, legend=True, pred_covar=None, ms=None, alpha=1)
