"""
The function in this file returns the parameters for the GCE NN.
"""
import os
import numpy as np
from GCE.utils import DotDict


def get_params(int_flag=0):
    """
    This function defines the parameters for the neural network-based analysis of gamma-ray photon-count maps.
    :param int_flag: integer flag that can be used to distinguish between different cases
    (useful if the same parameter file is used on several computers and only the filenames need to be changed, e.g.
    p_gen["fermi_root"] = <ROOT_1> if int_flag else <ROOT_2>).
    :return: dictionary containing all the parameters
    """
    p = DotDict()

    # General settings
    ###################################
    p_gen = DotDict()
    p_gen["data_root"] = "../data"  # data folder
    p_gen["fermi_root"] = os.path.join(p_gen["data_root"], "fermi_data_573w")  # root folder containing Fermi data
    p_gen["template_maps_root"] = os.path.join(p_gen["data_root"], "Template_maps")  # folder for template maps
    p_gen["combined_maps_root"] = os.path.join(p_gen["data_root"], "Combined_maps")  # folder for combined maps
    p_gen["models_root"] = "../models"  # folder for models
    # NOTE: the Fermi root folder should have a subfolder "fermi_data_<NSIDE>", where <NSIDE> = p["nside"] (see below)
    p["gen"] = p_gen

    # Data settings
    ###################################
    p_data = DotDict()
    p_data["outer_rad"] = 25.0  # outer ROI radius (in degrees, Galactic Center is at 0)
    p_data["inner_band"] = 2.0  # latitudes |b| < this value will be masked
    p_data["leakage_delta"] = 0.75  # for point-source map creation: PSs are sampled from a slightly larger ROI
    # (the margin is set in degrees here), allowing counts to leak into and out of the ROI.
    # NOTE: this is ignored if p_data["psf"] = False.
    p_data["mask_type"] = "3FGL"  # mask for known bright PSs: one of "None", "3FGL", "4FGL"
    p_data["nside"] = int(128)  # nside resolution parameter of the data
    p_data["exposure"] = "Fermi"  # one of "Fermi", "Fermi_mean", or constant integer
    p_data["psf"] = True  # if True: apply Fermi PSF to PS templates when generating PS maps
    # (see the function fermi_psf() in data_utils.py)
    p["data"] = p_data

    # Modeling settings
    ###################################
    p_mod = DotDict()
    # p_mod["models_P"] = ["dif_O_pibs", "dif_O_ic", "iso", "bub"]  # list of Poissonian templates
    p_mod["models_P"] = ["bub"]  # list of Poissonian templates
    # p_mod["models_PS"] = ["gce_12_PS", "thin_disk_PS"]  # list of PS templates
    p_mod["models_PS"] = ["gce_12_PS", "iso_PS"]  # list of PS templates
    # NOTE: point-source models use the same names as the Poissonian models, but with a trailing "_PS"!
    # p_mod["model_names_P"] = [r"diffuse $\pi^0$ + BS", "diffuse IC", "isotropic", r"$\it{Fermi}$ bubbles"]  # names: P
    p_mod["model_names_P"] = [r"$\it{Fermi}$ bubbles"]  # names: P
    p_mod["model_names_PS"] = ["GCE", "isotropic PS"]  # names: PS
    p["mod"] = p_mod

    # Template map settings (for training and testing maps)
    ###################################
    p_tt = DotDict()
    p_tt["data_name"] = "Example"  # name of data folder for the template maps
    p_tt["filename_base"] = "Maps"  # name basis of template map files
    p_tt["poisson_A_is_log"] = False  # is log10(A) rather than A specified for the Poissonian templates in prior_dict?
    p_tt["n_chunk"] = int(100)  # number of chunks to compute per job
    p_tt["n_sim_per_chunk"] = int(50)  # number of simulations per chunk and per model (one output file per chunk)
    # NOTE: the total number of maps for each template will be "n_chunk" * "n_sim_per_chunk" (* # jobs)
    p_tt["add_two_temps_PS"] = ["iso_PS"]  # list of PS templates for which TWICE the number of maps will be generated.
    # Later, these maps can be added pairwise, modeling two distinct populations.

    # Prior settings
    ###################################
    prior_dict = DotDict()

    # Poissonian templates: normalization A
    prior_dict["dif_O_pibs"] = np.asarray([7.0, 14.0])
    prior_dict["dif_O_ic"] = np.asarray([4.0, 9.0])
    prior_dict["iso"] = np.asarray([0.0, 2.0])
    prior_dict["bub"] = np.asarray([0.0, 2.0])
    prior_dict["gce_12"] = np.asarray([0.0, 2.5])
    prior_dict["dif"] = np.asarray([8.0, 18.0])  # p6v11 contains both contributions (pibs + ic)

    # The Poissonian prior values above correspond to nside = 128, scale by npix ratio (where npix = 12 * nside^2)
    if p_data["nside"] != 128:
        for key in prior_dict:
            prior_dict[key][0] /= (p_data["nside"] // 128) ** 2
            prior_dict[key][1] /= (p_data["nside"] // 128) ** 2

    # Priors for PS templates: SCDs modeled as skew normal distributions here
    # TODO: Note: changed upper limit -07 -> -08 for testing!
    prior_dict["gce_12_PS"] = {"mean_exp": [-12, -9], "var_exp": 0.25, "skew_std": 3.0,
                               "flux_lims": [0, 1.4e-08], "flux_log": False, "enforce_upper_flux": True}
    prior_dict["thin_disk_PS"] = {"mean_exp": [-12, -9], "var_exp": 0.25, "skew_std": 3.0,
                                  "flux_lims": [0, 2.8e-08], "flux_log": False, "enforce_upper_flux": True}
    prior_dict["iso_PS"] = {"mean_exp": [-12, -9], "var_exp": 0.25, "skew_std": 3.0,
                            "flux_lims": [0, 1.4e-08], "flux_log": False, "enforce_upper_flux": True}
    # Dict keys determine mean, variance, skew, total expected flux, and whether the log of flux_lims is specified.
    # If "enforce_upper_flux" is true: re-draw PS population if the sampled total flux exceeds upper prior limit
    # NOTE: if two template maps are summed up later, the upper flux limits should be HALF of the max. expected flux!
    p_tt["priors"] = prior_dict
    p["tt"] = p_tt

    # Settings for combining template maps
    ###################################
    p_comb = DotDict()
    p_comb["data_name"] = "Example_comb"  # name of data folder for the combined maps
    p_comb["filename_base"] = "Maps"  # name basis of combined map files
    p_comb["N_val"] = 2  # number of files for the validation data set
    p_comb["N_test"] = 2  # number of files for the testing data set
    # the remaining files will be used as training data

    # SCD histogram settings
    p_comb["hist_templates"] = ["gce_12_PS", "iso_PS"]  # list of templates for which histograms shall be saved
    p_comb["do_dNdF"] = True  # save histograms of dNdFs?
    p_comb["do_counts_per_PS"] = False  # save histograms of counts per PS?
    p_comb["do_counts_per_pix"] = False   # save histograms of counts per pixel (before applying the PSF)?
    p_comb["bins_dNdF"] = np.asarray([-np.infty] + list(np.logspace(-12.5, -7.0, 21)) + [np.infty])  # bins for SCD
    p_comb["power_of_F_dNdF"] = 1  # power of F to multiply dN/dF with.
    # NOTE: for log-spaced flux bins, we actually consider F dN/dlogF  ~  F^2 dN/dF
    p_comb["bins_counts_per_PS"] = np.asarray(list(np.linspace(0, 60, 21) - 0.5) + [np.infty])  # bins
    p_comb["bins_counts_per_pix"] = np.asarray(list(np.linspace(0, 60, 21) - 0.5) + [np.infty])  # bins
    p_comb["combine_without_PSF"] = False  # if True: combine template maps before PSF was applied
    p["comb"] = p_comb

    # Neural network settings
    ###################################
    p_nn = DotDict()
    p_nn["run_name"] = "Training_1"  # name of training run
    p_nn["NN_type"] = "CNN"  # type of NN: only "CNN" implemented so far
    p_nn["remove_exp"] = True  # remove exposure correction (work in terms of FLUX vs COUNTS)

    # Neural network architecture (for both FF and SCD submodels)
    ###################################
    p_arch = DotDict()
    p_arch['nsides'] = [128, 64, 32, 16, 8, 4, 2, 1]  # list containing nside hierarchy for a forward pass though the NN
    p_arch['F'] = [32, 64, 128, 256, 256, 256, 256]  # graph-convolutional layers: number of feature maps for each layer
    p_arch['M'] = [2048, 512]  # hidden fully-connected layers: output dimensionalities
    # NOTE: This should NOT include final fully-connected layer whose output dimension will automatically be computed
    p_arch['K'] = [5] * len(p_arch['F'])  # polynomial orders for the graph convolutions
    p_arch['is_resnet'] = [False] * len(p_arch['F'])  # use ResNet blocks instead of standard graph-convolutions
    p_arch['batch_norm'] = [1] * len(p_arch['F']) + [0] * len(p_arch['M'])  # batch (1) / instance (2) normalization
    # Note: should be specified for conv. layers and fully-connected layers, so len = len(...['F']) + len(...['M')
    p_arch['conv'] = 'chebyshev5'  # graph convolution: "chebyshev5" or "monomials"
    p_arch['pool'] = 'max'  # pooling: 'max' or 'avg'
    p_arch['append_tot_counts'] = True  # if p_nn["rel_counts"]: append total counts to input for first FC layer?
    p_arch['activation'] = 'relu'  # non-linearity: relu, elu, leaky_relu, etc.
    p_nn["arch"] = p_arch

    # Flux fractions
    ###################################
    p_ff = DotDict()
    p_ff["return_ff"] = True  # main switch for flux fraction estimation
    p_ff["alea_covar"] = False  # if True: estimate aleatoric uncertainty covariance matrix
    p_ff["alea_var"] = True  # if True: estimate aleatoric uncertainty variances, no correlations
    p_ff["rel_counts"] = True  # scale the pixel values by the total number of counts in the map?
    p_ff["last_act"] = "softmax"  # last activation function yielding the flux fraction mean estimates
    # "softmax" or "normalized_softplus"
    p_nn["ff"] = p_ff

    # SCD histograms (note: make sure that the desired histogram was saved when combining the template maps, see p_comb)
    ###################################
    p_hist = DotDict()
    p_hist["return_hist"] = True  # main switch for SCD histogram estimation
    p_hist["hist_templates"] = ["gce_12_PS", "iso_PS"]  # list of PS templates with histogram
    # NOTE: this must be subset of the templates for which the histograms were saved, given by p_comb["hist_templates")
    p_hist["last_act"] = "normalized_softplus"  # last activation function yielding the SCD histogram,
    # "softmax" or "normalized_softplus"
    p_hist["calculate_residual"] = True  # calculate FF residual and feed as an additional input to the brightness part
    p_hist["rel_counts"] = True  # feed relative counts (and normalized residual) to histogram part of the NN?
    p_hist["which_histogram"] = "dNdF"  # "dNdF", "counts_per_PS", or "counts_per_pix"
    p_hist["log_spaced_bins"] = True  # are the bins logarithmically spaced (otherwise: linear spacing is assumed)?
    p_nn["hist"] = p_hist

    # Training data selection settings (can be used to bias training data fed to the NN)
    ###################################
    p_cond = DotDict()
    p_cond["cond_on_training_data_str"] = "lambda x: x.sum() > 0"  # None or str with a lambda fct. of a single map that
    # evaluates to True or False, e.g. "lambda x: x.sum() < 1000". NOTE: 'eval' will be used to evaluate this expression
    p_cond["cond_on_training_labels_str"] = None  # None or str containing a lambda fct. of a single label
    # NOTE: if histogram estimation is activated, label[0] contains flux fractions, label[1] contains SCD histograms
    p_cond["prob_for_conditions"] = 1.0  # with this probability the conditions will be imposed for a given training map
    # (the resulting proportion of samples satisfying this condition will generally be higher because the condition
    # might be satisfied although it was not enforced!)
    p_nn["cond"] = p_cond
    p["nn"] = p_nn

    # Training settings
    ###################################
    p_train = DotDict()
    # NOTE: the batch sizes specified below set the GLOBAL batch size.
    # For example, setting p_train['batch_size'] = 256 yields n_batch = 64 on each GPU when using 4 GPUs.
    p_train['num_steps'] = 500  # number of steps to do (total number of maps shown is num_steps * batch_size)
    p_train['batch_size'] = 16  # number of samples per training batch. Should be a power of 2 for greater speed
    p_train['batch_size_val'] = 16  # number of samples per validation batch
    p_train['prefetch_batch_buffer'] = 5  # number of batches to prefetch for training data
    p_train['prefetch_batch_buffer_val'] = 5  # number of batches to prefetch for validation data
    p_train['eval_frequency'] = 50  # frequency of model evaluations during training (influences training time!)
    p_train['scheduler'] = 'ExponentialDecay'  # learning rate scheduler
    p_train['scheduler_dict'] = {"initial_learning_rate": 5e-4, "decay_steps": 1, "decay_rate": 1 - 0.00015,
                                 "staircase": False}  # scheduler settings
    p_train['optimizer'] = "Adam"  # optimizer
    p_train['optimizer_dict'] = {"beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8}  # optimizer settings
    p_train['ff_loss'] = 'l2'  # loss function for the flux fraction estimation
    # one of "l1" or "l2" (also includes normal max. likelihood loss for alea_covar / alea_var == True)
    p_train['hist_loss'] = 'EMPL'  # loss function for the SCD histogram estimation
    # one of: "l1", "l2", "EM1", "EM2", "CJS" (cumulative Jensen-Shannon), "x-ent", "EMPL" (Earth Mover's Pinball Loss)
    p_train['hist_pinball_smoothing'] = 0.001  # if > 0.0: take smoothed pinball loss (for example: 0.001)
    p_train['ff_train_metrics'] = ["l2", "l1", "x-ent"]  # flux fraction metrics for tensorboard
    p_train['hist_train_metrics'] = ["l2", "l1", "x-ent"]  # SCD histogram metrics for tensorboard
    p["train"] = p_train

    # Plot settings
    ###################################
    p_plot = DotDict()
    p_plot["colors_P"] = ['#37c837']  # plot colors for the Poissonian models
    p_plot["colors_PS"] = ['deepskyblue', 'k']  # plot colors for the point-source models
    p["plot"] = p_plot

    # Debugging settings (only booleans allowed)
    ###################################
    p_db = DotDict()
    p_db["deactivate_poiss_scatter_for_P"] = False  # if True: no Poissonian scatter for the Poissonian template maps
    # (can be used e.g. to test the residual layer: in this case, only PSs should remain if FFs are correct)
    p_db["chatterbox_generators"] = False  # if True: activate verbose generator output

    if np.any([k for k in p_db.values()]):
        p["db"] = p_db
        print("=== DEBUGGING MODE ===")

    # Return parameter dictionary
    return p
