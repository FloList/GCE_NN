"""
The function in this file returns the parameter for the GCE CNN with pre-generated data
(see arXiv:2006.12504).
"""
import numpy as np
import healpy as hp
import tensorflow as tf
from gce_utils import get_pixels, get_pixels_with_holes, get_template
from gce_data import PredictionParameters
from scipy import stats
from NPTFit.create_mask import make_mask_total


def get_params_CNN_pre_gen(HPC):
    params = dict()

    # General settings
    NAME = "Iso_maps_combined"
    NAME_ADD = "_IN"
    PROJECT_2D = False  # If True: don't use a graph CNN but project data to 2D image and use standard 2D CNN
                       # NOTE: experimental, many options (e.g. uncertainty estimation) are not implemented!
    OUTER_RAD = 25  # outer ROI radius to use
    RAD_SAFETY = 2 if PROJECT_2D else 1  # require that all the pixels within OUTER_RAD + RAD_SAFETY are within the ROI
    NSIDE = 256  # n_side for the data
    CONST_EXP = True  # is the exposure for the maps taken to be constant (e.g. Fermi mean exposure)?
    MODELS = ["iso_PS"]  # Templates
    MODEL_NAMES = ["Isotropic"]  # Template names
    EXP_NAME = NAME + NAME_ADD + "_pre_gen"  # name of experiment
    data_folder = '/scratch/u95/fl9575/GCE_v2/data/' + NAME if HPC else '/home/flo/PycharmProjects/GCE/data/' + NAME

    # GCE settings
    params["NN_type"] = "CNN"
    params["PRE_GEN"] = True
    params["models"] = MODELS
    params["model_names"] = MODEL_NAMES
    params["project_2d"] = PROJECT_2D
    params["outer_rad"] = OUTER_RAD
    params["rad_safety"] = RAD_SAFETY
    params["nside"] = NSIDE
    params["const_exp"] = CONST_EXP
    params["n_params"] = len(MODELS)  # length of labels = number of models (except for LLH loss, but this is treated separately)
    params["data_folder"] = data_folder
    params["unmask"] = False  # unmask = True -> run on ENTIRE sphere instead of fraction! MUCH SLOWER!
    params["ring2nest"] = False  # if input data is stored in RING format
    params["remove_exp"] = True  # remove exposure correction (work in terms of FLUX vs COUNTS)
    params["ROI_fermi"] = OUTER_RAD  # angle around the GC that contains data: Fermi data outside will be masked
    params["mask_type_fermi"] = "None"  # in the Fermi counts map: mask PSs? "3FGL", "4FGL", "None"
    params["rel_counts"] = True  # scale the pixel values by the total number of counts in the map?
    params["append_tot_counts"] = True  # if params["rel_counts"] = True: append total number of counts to input for first FC layer?
    params["aleatoric"] = False  # if True: estimate aleatoric uncertainties (including covariance matrix)
    params["alea_only_var"] = True  # if True: only estimate variances, no correlations
    params["alea_split"] = False  # if False: directly use log likelihood loss and fit means and covariances simultaneously
                                      # if True: step 1: train means only, using l2 loss.
                                      #          step 2: after training the means, fix the weights and only fit covariances
    params["alea_step"] = 1  # if splitting: step 1 or 2
    params["covar_scaling"] = 1.0  # scale aleatoric uncertainty covariance matrix for numerical stability
    params["epistemic"] = False  # estimate epistemic uncertainty
    params["epi_n_dropout"] = 10  # how many samples to evaluate to use for estimating epistemic uncertainty during training
    params["epi_p_init_min"] = 0.01  # min. initial dropout prob.
    params["epi_p_init_max"] = 0.01  # max. initial dropout prob.
    params["epi_prior_length_scale"] = 1.0  # epistemic prior length scale l (e.g. 1e-4)
    params["epi_n_training_samples"] = 6e5 if HPC else 2.2e4  # number of training samples (needed for Concrete Dropout priors)
    params["epi_dropout_output_layer"] = True  # do MC dropout for last FC layer?

    # Brightness histograms (note: data is already saved binned)
    params["gce_hist_templates"] = ["iso_PS"]  # list of PS templates with histogram
    params["gce_return_hist"] = True  # return histogram with GCE DM & PS distribution
    params["gce_only_hist"] = True  # if True: don't build NN for GCE count/flux fraction estimation
    params["gce_hist_loss"] = "CDF_pinball"  # implementations for now: "l1", "l2", "EM1", "EM2", "CJS" (cumul. Jensen-Shannon), "x-ent"
    params["gce_hist_act"] = "softmax"  # activation function for the GCE histogram output:
                                        # options are: "softmax" or "softplus" (normalised by sum(softplus))
    params["gce_hist_FF_weights_loss"] = False  # Weight the histogram loss with FF of the respective template (higher FF: counts more for loss)
    params["calculate_residual"] = False  # calculate FF residual and feed as an additional input to the brightness part
    params["gce_hist_rel_counts"] = True  # feed relative counts (and normalised residual) to histogram part of the NN?
    params["gce_hist_split"] = False  # split histogram prediction and FF prediction? 1) FFs, then 2) histograms
    params["gce_hist_step"] = 1  # 1) FFs, then 2) histograms

    # CDF Pinball loss options (WORK IN PROGRESS!)
    params["gce_hist_tau_dist"] = "uniform"  # "uniform", "arcsin" (places more weight on small and large quantiles)
    params["gce_hist_pinball_smoothing"] = 0.001  # if > 0.0: take smoothed pinball loss (for example: 0.001)

    params["gce_hist_lambda"] = 1.0  # histogram loss is multiplied by this number

    # Different histograms implementations (WORK IN PROGRESS!)
    params["which_histogram"] = 1  # 1: dNdF, 2: counts per PS, 3: counts per pix

    # Training data selection settings
    params["cond_on_training_data"] = None  # lambda function of a single map
    params["cond_on_training_labels"] = None  # lambda function of a flux contribution vector
    params["prob_for_conditions"] = 0.0  # impose these conditions with this probability
                                         # (the resulting proportion of samples fullfilling this condition will be higher!)

    # Template settings (if None: inference without using templates.)
    # NOTE: the following options are NOT implemented for PROJECT_2D!
    # NOTE: templates are assumed to be in RING ordering and will be converted to NESTED
    # No need to use template iso since it is homogeneous in space, so doesn't provide any information
    params["template_path"] = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data_573w' if HPC \
        else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data_573w'
    params["template_path"] += "/fermi_data_" + str(NSIDE)
    params["exp_name"] = "fermidata_exposure"  # name of exposure file

    # Set parameters for DeepSphere
    params['dir_name'] = EXP_NAME

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['new_weighting'] = True  # use new weighting scheme, with kernel width taken from https://openreview.net/pdf?id=B1e3OlStPB
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['last_act'] = 'softmax'   # Last activation: softmax (sum(labels) = 1, WORKS MUCH BETTER!) or sigmoid (enforce sum(fractions) = 1 using lambda)
                                    # note: this is ignored if loss is LLH_RATIO
    params['loss'] = 'None'        # "l2" loss for regression (also includes normal max. likelihood loss for aleatoric == True) or
                                    # "l1" loss for regression (also include Laplace max. likelihood loss for aleatoric == True)
                                    # "x-ent" (choose linear last_act!), or "None" if only estimating histogram
                                    # or: LLH_RATIO (see http://arxiv.org/abs/2011.13951).

    # LLH_RATIO options: register the parameters that shall be estimated here
    pp = PredictionParameters()
    for temp in MODELS:
        # 1) Poissonian templates: only A
        if "PS" not in temp:
            normalise_A = lambda a, lim_min, lim_max: (a - lim_min) / (lim_max - lim_min)
            unnormalise_z = lambda z, lim_min, lim_max: lim_min + z * (lim_max - lim_min)
            pp.register_param("A_" + temp, "A", temp, normalise_A, unnormalise_z, requires_prior_lims=True)
        # 2) Non-Poissonian templates:
        else:
            # Mean
            normalise_mean = lambda mu, lim_min, lim_max: (mu - lim_min) / (lim_max - lim_min)
            unnormalise_z = lambda z, lim_min, lim_max: lim_min + z * (lim_max - lim_min)
            pp.register_param("means_" + temp, "means", temp, normalise_mean, unnormalise_z, requires_prior_lims=True,
                              prior_lims_key="mean_exp")

            # Var: drawn from a chi2 distribution
            normalise_var = lambda var, var_exp: stats.chi2.cdf(var / var_exp, df=1)
            unnormalise_z = lambda z, var_exp: stats.chi2.ppf(z, df=1) * var_exp
            pp.register_param("vars_" + temp, "vars", temp, normalise_var, unnormalise_z, requires_prior_lims=True,
                              prior_lims_key="var_exp")

            # Skew: drawn from a normal distribution
            normalise_skew = lambda skew, skew_std: stats.norm.cdf(skew, loc=0, scale=skew_std)
            unnormalise_z = lambda z, skew_std: stats.norm.ppf(z, loc=0, scale=skew_std)
            pp.register_param("skew_" + temp, "skew", temp, normalise_skew, unnormalise_z, requires_prior_lims=True,
                              prior_lims_key="skew_std")

            # Total flux: drawn from a normal distribution
            normalise_tot_flux = lambda tot_flux, lim_min, lim_max: (tot_flux - lim_min) / (lim_max - lim_min)
            unnormalise_z = lambda z, lim_min, lim_max: lim_min + z * (lim_max - lim_min)
            pp.register_param("tot_flux_" + temp, "tot_flux", temp, normalise_tot_flux, unnormalise_z, requires_prior_lims=True,
                              prior_lims_key="flux_lims")


    params['pp'] = pp

    # Data transformation (for tf tensors, not np!)
    # NOTE: The data transformation is applied AFTER removing the exposure correction present in simulated & Fermi data
    if params["rel_counts"]:
        params["data_trafo"] = lambda data: data
    else:
        params["data_trafo"] = lambda data: data

    # Architecture.
    params['nsides'] = [256, 128, 64, 32, 16, 8, 4, 2, 1]  # number of nsides per layer (for PROJECT_2D: only the first value will be used)
    params['F'] = [32, 64, 128, 256, 256, 256, 256, 256]  # (Graph) convolutional layers: number of feature maps.
    params['M'] = [2048, 512, params["n_params"]]  # Fully connected layers: output dimensionalities.

    assert params['nsides'][0] == NSIDE, "NN nsides values don't match NSIDE! Check the NN architecture!"

    if PROJECT_2D:
        params['input_shape'] = [128, 128]
        params['K'] = [[5, 5]] * len(params['F'])  # Kernel sizes (list of 2D arrays for 2DCNN)
        params['p'] = [1] * len(params['F'])  # Strides for cnn2D
        params['pool_fac'] = [2] * len(params['F'])  # pooling factors for cnn2D: use EITHER stride OR pooling!`
    else:
        params['K'] = [5] * len(params['F'])  # Polynomial orders. (1D array for Deepsphere)
        params["glob_avg"] = False  # Global averaging layer (avg. over spatial dims.) before FC layers (makes only sense if not convolving down to 1 px)

    params["is_resnet"] = [False] * (len(params['F']) - 0)  # conv layer -> resnet block
    params["has_se"] = [False] * (len(params['F']) - 0)  # has squeeze-excitation?
    params['batch_norm'] = [2] * len(params['F'])  # Batch normalization (1) / instance normalisation (2)

    # Regularization.
    # NOTE: THIS ONLY HAS AN EFFECT IF EPISTEMIC UNCERTAINTY ESTIMATION VIA CONCRETE DROPOUT IS DISABLED!
    params['regularization'] = 0.0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1.0  # Percentage of neurons to keep. NOTE: this is only used if epistemic uncertainty is not estimated using Concrete Dropout
    params['lambda_penalty'] = 0.0  # Strength of penalty term that enforces the sum of the labels to be = 1 (only needed if last_act is not softmax), NOT RECOMMENDED!

    # Training.
    params['num_steps'] = 25000  # Number of steps to do (total number of maps shown is num_steps * batch_size)
    params['batch_size'] = 64 if HPC else 16  # Number of samples per training batch. Should be a power of 2 for greater speed.
    params['batch_size_val'] = 64 if HPC else 16  # Number of samples for validation
    params['prefetch_batch_buffer'] = 5  # Number of batches to prefetch
    params['prefetch_batch_buffer_val'] = 5  # Number of batches to prefetch
    params['eval_frequency'] = 250  # Frequency of model evaluations during training (influences training time).
    params['scheduler'] = lambda step: 5e-4 * tf.exp(-0.00015 * tf.cast(step, dtype=tf.float32))  # learning rate lamda fct. (e.g. 5e-4),
    params['optimizer'] = lambda lr: tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # If PROJECT_2D: always need to unmask!
    if PROJECT_2D:
        params["unmask"] = True

    # Determine pixels for each nsides
    # Choose mask slightly larger!
    # ROI = hp.reorder(make_mask_total(nside=params["nsides"][-1], mask_ring=True, inner=0, outer=OUTER_RAD + RAD_SAFETY),
    #                  r2n=True)
    # params["indexes"] = get_pixels(ROI, params["nsides"])

    # Set up the mask for the ROI: NEW: holes are allowed
    ENFORCE_ROI_NSIDE_1 = True
    nside_for_extended_mask = 1 if ENFORCE_ROI_NSIDE_1 else params["nsides"][-1]

    inner_band = 0
    outer_ring = params["outer_rad"]
    nside = params["nside"]
    ROI = make_mask_total(band_mask=True, band_mask_range=inner_band, mask_ring=True, inner=0,
                                        outer=outer_ring, nside=nside)
    if params["mask_type_fermi"] == "3FGL":
        ROI = (1 - (1 - ROI) * (1 - get_template(params["template_path"], "3FGL_mask"))).astype(bool)
    elif params["mask_type_fermi"] == "4FGL":
        ROI = (1 - (1 - ROI) * (1 - get_template(params["template_path"], "4FGL_mask"))).astype(bool)
    ROI = hp.reorder(ROI, r2n=True)

    ROI_extended = hp.reorder(make_mask_total(nside=nside_for_extended_mask, mask_ring=True, inner=0,
                                              outer=OUTER_RAD + RAD_SAFETY), r2n=True)
    params["indexes"] = get_pixels_with_holes(ROI, params["nsides"])
    params["indexes_extended"] = get_pixels(ROI_extended, params["nsides"])
    params["do_neighbor_scaling"] = False

    # Flags for pre-generated data
    params["estimate_var_fracs"] = False  # TODO: IMPLEMENT!
    params["model_vars"] = [None] * len(MODELS)  # not implemented (yet), might be added at some point
    params["is_log_A"] = False  # this information is needed for llh-ratio estimation to convert map parameters to FFs
                                # however, this information was initially not stored in the settings_dict, so set it
                                # manually here
    # Return parameters
    return params
