"""
The function in this file returns the parameter for the GCE CNN.
"""
import numpy as np
import healpy as hp
import tensorflow as tf
from gce_utils import get_pixels, get_template
from NPTFit.create_mask import make_mask_total


def get_params_CNN(HPC):

    # General settings
    NAME = "GCE_poisson_only"
    PROJECT_2D = 0  # 0: use a graph CNN
                    # 1: don't use a graph CNN but project data to 2D image and use standard 2D CNN
                    #    NOTE: experimental, many options (e.g. uncertainty estimation) are not implemented!

    OUTER_RAD = 25  # outer ROI radius to use
    RAD_SAFETY = 2 if PROJECT_2D else 1  # require that all the pixels within OUTER_RAD + RAD_SAFETY are within the ROI
    NSIDE = 128  # n_side for the data
    # MODELS = ["gce"]
    # MODEL_VARS = [["gce_12"]]
    # MODEL_NAMES = ["GCE"]
    MODELS = ["dif_pibs", "dif_ic", "iso", "bub", "gce"]  #, "gce_PS"]  #, "disk_PS"]  # Templates
    MODEL_VARS = [["dif_O_pibs"], ["dif_O_ic"], ["iso"], ["bub"], ["gce_12"]]   #, ["gce_12_PS"]]  #, ["disk_PS"]]  # Variants of each template
    MODEL_NAMES = [r"diffuse $\pi^0$ + BS", "diffuse IC", "isotropic", r"$\it{Fermi}$ bubbles", "GCE"]  #, "GCE PS"]  #, "disk PS"]  # Template names
    EXP_NAME = NAME + "_onthefly"  # name of experiment
    fermi_path = "/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data" if HPC else "/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data"
    data_folder = None
    CONST_EXP = True  # take constant mean Fermi exposure for training data
    MASK_3FGL = False  # mask 3FLG sources at 95% containment radius
    BAND_MASK_RANGE = 0  # mask range for |b|
    if MASK_3FGL:
        assert NSIDE == 128, "When masking 3FGL, NSIDE = 128 is required!"
        print("3FGL SOURCES WILL BE MASKED!")
        pscmask = get_template(fermi_path, "3FGL_mask")
        MASK = (1 - (1 - make_mask_total(band_mask=True, band_mask_range=BAND_MASK_RANGE, mask_ring=True, inner=0, outer=OUTER_RAD)) * (1 - pscmask))
    else:
        print("3FGL SOURCES WILL NOT BE MASKED!")
        MASK = make_mask_total(band_mask=True, band_mask_range=BAND_MASK_RANGE, mask_ring=True, inner=0, outer=OUTER_RAD, nside=NSIDE)

    # Prior dict
    prior_dict = dict()
    # prior_dict["dif_pibs"] = prior_dict["dif_ic"] = [0, 2]   # LOG
    # prior_dict["bub"] = prior_dict["iso"] = [-3, 2]  # LOG
    # prior_dict["gce"] = [-np.infty] * 2  # LOG
    prior_dict["dif_pibs"] = prior_dict["dif_ic"] = [2, 10]  # this gives ~15-70 %  LINEAR
    prior_dict["bub"] = prior_dict["iso"] = [0, 2.5]  # this gives ~0-20 %  LINEAR
    prior_dict["gce"] = [0, 2.5] * 2

    # PS templates: load from files (model name must end with "_PS")
    # NOTE: make sure that the correct settings were used for the PS map generation!
    # Maps are expected to be saved in RING format!
    # Also: each PS template must correspond to exactly ONE model variant
    if CONST_EXP:
        folder_maps_PS = '/scratch/u95/fl9575/GCE_v2/data/GCE_maps_hist_const_exp' if HPC else '/home/flo/PycharmProjects/GCE/data/GCE_maps_hist_const_exp'
    else:
        folder_maps_PS = '/scratch/u95/fl9575/GCE_v2/data/GCE_maps_hist' if HPC else '/home/flo/PycharmProjects/GCE/data/GCE_maps_hist'

    # Params dict
    params = dict()
    params["NN_type"] = "CNN"
    params["models"] = MODELS
    params["model_names"] = MODEL_NAMES
    params["model_vars"] = MODEL_VARS
    params["project_2d"] = PROJECT_2D
    params["outer_rad"] = OUTER_RAD
    params["rad_safety"] = RAD_SAFETY
    params["nside"] = NSIDE
    params["prior_dict"] = prior_dict
    params["log_priors"] = False
    params["folder_maps_PS"] = folder_maps_PS
    params["test_fraction_PS"] = 1./15  # fraction of PS maps for test set
    params["no_PSF"] = False  # if True: take 2nd channel that contains the maps without PSF smoothing!
    params["mask"] = MASK
    params["const_exp"] = CONST_EXP
    params["remove_exp"] = True  # remove exposure correction (work in terms of FLUX vs COUNTS)
    params["n_params"] = len(MODELS)  # length of labels = number of models
    params["data_folder"] = data_folder
    params["unmask"] = False  # unmask = True -> run on ENTIRE sphere instead of fraction! MUCH SLOWER!
    params["ROI_fermi"] = OUTER_RAD  # angle around the GC that contains data: Fermi data outside will be masked
    params["mask_3FGL_fermi"] = MASK_3FGL  # in the Fermi counts map: mask the 3FGL sources?
    params["rel_counts"] = True  # scale the pixel values by the total number of counts in the map?
    params["append_tot_counts"] = True  # if params["rel_counts"] = True: append total number of counts to input for first FC layer?
    params["aleatoric"] = False  # True if not PROJECT_2D else False  # if True: estimate aleatoric uncertainties (including covariance matrix)
    params["alea_only_var"] = True  # if True: only estimate variances, no correlations
    params["alea_split"] = False  # if False: directly use log likelihood loss and fit means and covariances simultaneously
                                      # if True: step 1: train means only, using l2 loss.
                                      #          step 2: after training the means, fix the weights and only fit covariances
    params["alea_step"] = 1  # if splitting: step 1 or 2
    params["covar_scaling"] = 1.0  # scale aleatoric uncertainty covariance matrix for numerical stability
    params["epistemic"] = False  # True if not PROJECT_2D else False  # estimate epistemic uncertainty
    params["epi_n_dropout"] = 10  # how many samples to evaluate to use for estimating epistemic uncertainty during training
    params["epi_p_init_min"] = 0.01  # min. initial dropout prob.
    params["epi_p_init_max"] = 0.01  # max. initial dropout prob.
    params["epi_prior_length_scale"] = 1e-4  # epistemic prior length scale l (e.g. 1e-4)
    params["epi_n_training_samples"] = 6e5 if HPC else 2.2e4  # number of training samples (needed for Concrete Dropout priors)
    params["epi_dropout_output_layer"] = True  # do MC dropout for last FC layer?

    # Combine GCE DM + GCE PS (WORK IN PROGRESS!)
    params["combine_gce"] = False  # GCE PS will be added to GCE DM and the two will be treated as a single template
    params["gce_PS_var"] = ["gce_12_PS"]  # variant name for the GCE PSs to use
    params["gce_return_hist"] = False  # return histogram with GCE DM & PS distribution
    params["gce_hist_separate"] = False  # use separate ConvBlock architecture for the GCE histograms
    params["gce_only_hist"] = False  # if True: don't build NN for GCE count/flux fraction estimation
    params["gce_hist_bins"] = np.asarray([-np.infty] + list(np.logspace(-12.5, -6, 41)) + [np.infty])
    # params["gce_hist_bins"] = np.asarray(list(np.linspace(0, 60, 21) - 0.5) + [np.infty])  # edges of histogram bins (e.g.: mod. Fibonacci)
    # params["gce_hist_bins"] = np.asarray([-0.5, 0.5, np.infty])
                                # NOTE: depending on "which_histogram": in flux space or count space!
    params["gce_hist_loss"] = "l1"  # implementations for now: "l1", "l2", "EM1", "EM2", "CJS" (cumul. Jensen-Shannon), "x-ent"
    params["gce_hist_act"] = "softmax"  # activation function for the GCE histogram output:
                                        # options are: "softmax" or "softplus" (normalised by sum(softplus))

    params["gce_hist_uncertainties"] = False  # uncertainties for histogram
    params["gce_hist_uncertainties_eps"] = 1e-10  # min. uncertainty variance is given by np.sqrt(this number)
    params["gce_hist_corr"] = False  # correlated uncertainties? (only available for "l2" loss!)
    params["gce_hist_corr_init"] = np.asarray([0.0, 0.0, 0.0])  # sub/super diagonals: initial correlations for entries that shall be trained
                                     # e.g. np.asarray([0.5, 0.2, 0.0]) means: corr. betweens bins with distance 3 are considered

    # CDF Pinball loss options (WORK IN PROGRESS!)
    params["gce_hist_tau_dist"] = "uniform"  # "uniform", "arcsin" (places more weight on small and large quantiles)
    params["gce_hist_pinball_smoothing"] = 0.001  # if > 0.0: take smoothed pinball loss (for example: 0.001)

    params["gce_hist_lambda"] = 1.0  # histogram loss is multiplied by this number
    params["gce_hist_weight_dm_half"] = False  # weight DM not as a single bin but such that it accounts as much as all the other PSs bins together
                                              # NOTE: only has an effect for "l1" or "l2" hist_loss
    # Different histograms implementations (WORK IN PROGRESS!)
    params["which_histogram"] = 1
    # List:
    # -1: dNdF: old implementation where the dN/dF is a power law - the histogram is generated from A, S, and n
    # 1: dNdF
    # 2: counts per PS
    # 3: counts per pixel without PSF
    params["power_of_F"] = 1  # power of F to multiply dN/dF with (only for histogram option 1)

    # FFJORD (WORK IN PROGRESS!)
    # Uncertainty calibration: all the NN weights will be frozen except for the last layer and the weights of the calibration method
    params["build_ffjord"] = False  # build FFJORD non-Gaussianian calibration: NOTE: currently requires l2-type loss!
    params["use_ffjord"] = False  # use FFJORD (after the NN has been trained with a diagonal Gaussian uncertainty)
    ffjord_dict = dict()
    ffjord_dict["stacked_ffjords"] = 1  # number of ffjords to use
    ffjord_dict["n_hidden_ffjord"] = 128  # number of hidden neurons for FFJORD
    ffjord_dict["n_layers_ffjord"] = 3  # number of layers for FFJORD
    ffjord_dict["solver_tol_ffjord"] = 1e-5  # tolerance for FFJORD ODE solver
    ffjord_dict["trace_exact_ffjord"] = False  # use exact trace formula instead of Hutchinson trace estimator
    ffjord_dict["std_if_not_given_ffjord"] = 0.1  # start with this STD if the NN has been trained without (aleat.) uncertainty estimation
    ffjord_dict["n_sample_ffjord"] = 128  # number of FFJORD samples to draw in one go
    ffjord_dict["activation_ffjord"] = "softplus"  # activation function for FFJORD
    ffjord_dict["activation_last_ffjord"] = "linear"  # activation function for FFJORD output layer
    params["ffjord_dict"] = ffjord_dict

    # Gaussian mixture
    params["gaussian_mixture"] = False  # model aleatoric uncertainties as a mixture of Gaussians
    params["n_gaussians"] = 3  # number of Gaussians to use for each template
    params["truncate_gaussians"] = True  # truncate Gaussians?
    params["distribution_strategy"] = "ratio"  # "diff" or "ratio": determines whether normalisation of the FFs leaves
                                              # the relative differences between the means constant or the ratios
    params["deactivate_hists"] = True  # deactivate tensorflow histograms for tensorboard

    # Training data selection settings
    params["cond_on_training_data"] = lambda x: x.sum() > 0  # lambda function of a single map
    params["cond_on_training_labels"] = None  # lambda function of a flux contribution vector
    params["cond_on_training_n_phot"] = None  # lambda function of a photon count list (per PS)
    params["prob_for_conditions"] = 1.0  # impose these conditions with this probability
                                         # (the resulting proportion of samples fullfilling this condition will be higher!)

    # Template settings
    # NOTE: the following options are NOT implemented for PROJECT_2D!
    # NOTE: templates are assumed to be in RING ordering and will be converted to NESTED
    # No need to use template iso since it is homogeneous in space, so doesn't provide any information
    params["template_path"] = fermi_path
    params["template_names"] = None  # set this to None
    params["exp_name"] = "fermidata_exposure"  # name of exposure file
    params["template_blocks"] = False  # if True: split up convolutional part of the NN in different blocks for different templates
                                      # otherwise: just append map * template as additional channels for the first convolutional layer
    params["append_map"] = False  # if True: append the full map (unconvolved) to the output of the conv. layers before FC layers

    # Set parameters for DeepSphere
    params['dir_name'] = EXP_NAME

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['new_weighting'] = True  # use new weighting scheme, with kernel width taken from https://openreview.net/pdf?id=B1e3OlStPB
    params['pool'] = 'max'  # Pooling: max or average. (for EfficientNet: flatten is also possible)
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['last_act'] = 'softmax'  # Last activation: softmax (sum(labels) = 1, WORKS MUCH BETTER!) or sigmoid (enforce sum(fractions) = 1 using lambda)
    params['statistics'] = None  # Set to None
    params['loss'] = 'l2'  # "l2" loss for regression (also includes normal max. likelihood loss for aleatoric == True) or
                                  # "l1" loss for regression (also include Laplace max. likelihood loss for aleatoric == True)
                                  # "x-ent" (choose linear last_act!), or "None" if only estimating histogram

    # Data transformation (for tf tensors, not np!)
    # NOTE: The data transformation is applied AFTER removing the exposure correction present in simulated & Fermi data
    if params["rel_counts"]:
        params["data_trafo"] = lambda data: data
    else:
        params["data_trafo"] = lambda data: data

    # Architecture.
    params['nsides'] = [128, 64, 32, 16, 8, 4, 2, 1]  # number of nsides per layer (for PROJECT_2D: only the first value will be used)
    params['F'] = [32, 64, 128, 256, 256, 256, 256]  # (Graph) convolutional layers: number of feature maps.
    params['M'] = [2048, 512, params["n_params"]]  # Fully connected layers: output dimensionalities.

    assert params['nsides'][0] == NSIDE, "NN nsides values don't match NSIDE! Check the NN architecture!"

    if PROJECT_2D == 1:  # Standard 2D CNN
        params['input_shape'] = [128, 128]
        params['K'] = [[5, 5]] * len(params['F'])  # Kernel sizes (list of 2D arrays for 2DCNN)
        params['p'] = [1] * len(params['F'])  # Strides for cnn2D
        params['pool_fac'] = [2] * len(params['F'])  # pooling factors for cnn2D: use EITHER stride OR pooling!`
    elif PROJECT_2D == 0:  # GCNN
        params['K'] = [5] * len(params['F'])  # Polynomial orders. (1D array for Deepsphere)
        params['glob_avg'] = False  # Global averaging layer (avg. over spatial dims.) before FC layers (makes only sense if not convolving down to 1 px)
    else:
        raise NotImplementedError

    params["is_resnet"] = [False] * (len(params['F']) - 0)  # conv layer -> resnet block
    params["has_se"] = [False] * (len(params['F']) - 0)  # has squeeze-excitation?
    params['batch_norm'] = [1] * len(params['F'])  # Batch normalization (1) / instance normalisation (2)

    # Estimation of the different template variant fractions
    params["estimate_var_fracs"] = False  # Estimate fractions of template variants
    params["glob_loss_lambda"] = 1.0  # factor to multiply the L1 loss for the template variants with

    # Regularization.
    # NOTE: THIS ONLY HAS AN EFFECT IF EPISTEMIC UNCERTAINTY ESTIMATION VIA CONCRETE DROPOUT IS DISABLED!
    params['regularization'] = 0.0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1.0  # Percentage of neurons to keep. NOTE: this is only used if epistemic uncertainty is not estimated using Concrete Dropout
    params['lambda_penalty'] = 0.0  # Strength of penalty term that enforces the sum of the labels to be = 1 (only needed if last_act is not softmax), NOT RECOMMENDED!

    # Training.
    params['num_steps'] = 25000  # Number of steps to do (total number of maps shown is num_steps * batch_size)
    params['batch_size'] = 64 if HPC else 16  # Number of samples per training batch. Should be a power of 2 for greater speed.
    params['batch_size_val'] = 64 if HPC else 16  # Number of samples for validation
    params['prefetch_batch_buffer'] = 3  # Number of batches to prefetch
    params['prefetch_batch_buffer_val'] = 3  # Number of batches to prefetch
    params['eval_frequency'] = 50  # Frequency of model evaluations during training (influences training time).
    params['scheduler'] = lambda step: 5e-4 * tf.exp(-0.00025 * tf.cast(step, dtype=tf.float32))  # learning rate lamda fct. (e.g. 5e-4)
    # params['scheduler'] = lambda step: 2e-5 * tf.ones_like(tf.cast(step, dtype=tf.float32))
                         # NOTE: function of GLOBAL step (continues when resuming training)!
                         # To reset lr when resuming, correct by setting step <- step - step_0
    params['optimizer'] = lambda lr: tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    # params['optimizer'] = lambda lr: tf.compat.v1.train.RMSPropOptimizer(lr)

    # If PROJECT_2D: always need to unmask!
    if PROJECT_2D:
        params["unmask"] = True

    # Determine pixels for each nsides
    # Choose mask slightly larger!
    ENFORCE_ROI_NSIDE_1 = True
    nside_for_mask = 1 if ENFORCE_ROI_NSIDE_1 else params["nsides"][-1]
    ROI = hp.reorder(make_mask_total(nside=nside_for_mask, mask_ring=True, inner=0, outer=OUTER_RAD + RAD_SAFETY),
                     r2n=True)
    params["indexes"] = get_pixels(ROI, params["nsides"])

    # DEBUG:
    params["profile"] = False

    # Return parameters
    return params
