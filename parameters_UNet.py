"""
The function in this file returns the parameter for the GCE U-Net.
"""
import healpy as hp
import tensorflow as tf
from gce_utils import get_pixels, get_template
from NPTFit.create_mask import make_mask_total


def get_params_UNet(HPC):

    # General settings
    NAME = "GCE_and_background"
    OUTER_RAD = 25  # outer ROI radius to use
    RAD_SAFETY = 1  # require that all the pixels within OUTER_RAD + RAD_SAFETY are within the ROI
    NSIDE = 128  # n_side for the data
    MODELS = ["dif_pibs", "dif_ic", "iso", "bub", "gce", "gce_PS", "disk_PS"]  # Templates
    MODEL_VARS = [["dif_O_pibs", "dif_A_pibs", "dif_F_pibs"], ["dif_O_ic", "dif_A_ic", "dif_F_ic"], ["iso"], ["bub"], ["gce_12"], ["gce_12_PS"], ["disk_PS"]]  # Variants of each template
    MODEL_NAMES = [r"diffuse $\pi^0$ + BS", "diffuse IC", "isotropic", r"$\it{Fermi}$ bubbles", "GCE DM", "GCE PS", "disk PS"]  # Template names
    PROJECT_2D = False  # must be set to False for U-Net!
    EXP_NAME = NAME + "_UNet"  # name of experiment
    fermi_path = "/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data" if HPC else "/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data"
    data_folder = None
    CONST_EXP = False  # take constant mean Fermi exposure for training data
    MASK_3FGL = False  # mask 3FLG sources at 95% containment radius
    if MASK_3FGL:
        assert NSIDE == 128, "When masking 3FGL, NSIDE = 128 is required!"
        print("3FGL SOURCES WILL BE MASKED!")
        pscmask = get_template(fermi_path, "3FGL_mask")
        MASK = (1 - (1 - make_mask_total(band_mask=True, band_mask_range=2, mask_ring=True, inner=0,
                                         outer=OUTER_RAD)) * (1 - pscmask))
    else:
        print("3FGL SOURCES WILL NOT BE MASKED!")
        MASK = make_mask_total(band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=OUTER_RAD, nside=NSIDE)

    # Prior dict
    prior_dict = dict()
    prior_dict["dif_pibs"] = prior_dict["dif_ic"] = [0, 2]
    prior_dict["bub"] = prior_dict["iso"] = prior_dict["gce"] = [-3, 2]

    # PS templates: load from files (model name must end with "_PS")
    # NOTE: make sure that the correct settings were used for the PS map generation!
    # Maps are expected to be saved in RING format!
    # Also: each PS template must correspond to exactly ONE model variant
    if CONST_EXP:
        folder_maps_PS = '/scratch/u95/fl9575/GCE_v2/data/GCE_maps_const_exp' if HPC else '/home/flo/PycharmProjects/GCE/data/GCE_maps_const_exp'
    else:
        folder_maps_PS = '/scratch/u95/fl9575/GCE_v2/data/GCE_maps' if HPC else '/home/flo/PycharmProjects/GCE/data/GCE_maps'

    # Params dict
    params = dict()
    params["NN_type"] = "U-Net"
    params["models"] = MODELS
    params["model_names"] = MODEL_NAMES
    params["model_vars"] = MODEL_VARS
    params["project_2d"] = PROJECT_2D
    params["outer_rad"] = OUTER_RAD
    params["rad_safety"] = RAD_SAFETY
    params["nside"] = NSIDE
    params["prior_dict"] = prior_dict
    params["log_priors"] = True
    params["folder_maps_PS"] = folder_maps_PS
    params["test_fraction_PS"] = 1./15  # fraction of PS maps for test set
    params["mask"] = MASK
    params["const_exp"] = CONST_EXP
    params["n_params"] = len(MODELS)  # labels have dimension n_batch x n_params x pixels
    params["data_folder"] = data_folder
    params["estimate_templates"] = False  # if True: estimate TEMPLATES, otherwise Poissonian REALIZATIONS!
    params["ROI_fermi"] = OUTER_RAD  # angle around the GC that contains data: Fermi data outside will be masked
    params["mask_3FGL_fermi"] = MASK_3FGL  # in the Fermi counts map: mask the 3FGL sources?
    params["rel_counts"] = True  # scale the pixel values by the total number of counts in the map?
    params["aleatoric"] = False  # if True: estimate aleatoric uncertainties (including covariance matrix)  # NOT IMPLEMENTED FOR U-NET YET
    params["alea_only_var"] = False  # if True: only estimate variances, no correlations  # NOT IMPLEMENTED FOR U-NET YET
    params["alea_split"] = False  # if False: directly use log likelihood loss and fit means and covariances simultaneously  # NOT IMPLEMENTED FOR U-NET YET
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

    # Template settings
    # NOTE: the following options are NOT implemented for PROJECT_2D!
    # NOTE: templates are assumed to be in RING ordering and will be converted to NESTED
    # No need to use template iso since it is homogeneous in space, so doesn't provide any information
    params["template_path"] = "/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data" if HPC else "/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data"
    params["exp_name"] = "fermidata_exposure"  # name of exposure file

    # Set parameters for DeepSphere
    params['dir_name'] = EXP_NAME

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['new_weighting'] = True  # use new weighting scheme, with kernel width taken from https://github.com/deepsphere/paper-deepsphere-iclr2020
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation_enc'] = 'leaky_relu'  # Non-linearity for ENCODING path: relu, elu, leaky_relu, softmax, tanh, etc.
    params['activation_dec'] = 'relu'  # Non-linearity for DECODING path: relu, elu, leaky_relu, softmax, tanh, etc.

    params['last_act'] = 'softmax'  # Last activation: softmax (sum(labels) = 1, WORKS MUCH BETTER!)
    params['statistics'] = None  # Set to None
    params['loss'] = 'l1'  # l2 loss for regression or l1 loss for regression  # todo: cross-entropy?

    # Data transformation (for tf tensors, not np!)
    # NOTE: The data transformation is applied AFTER removing the exposure correction present in simulated & Fermi data
    if params["rel_counts"]:
        params["data_trafo"] = lambda data: data
    else:
        params["data_trafo"] = lambda data: data

    # Architecture.
    params['nsides'] = [128, 64, 32, 16, 8, 4, 2, 1]  # number of nsides per layer (for Laplacian calculation)
    params['F_0'] = 8  # number of channels after the first conv. layer
    params['F_max'] = 16  # number of channels will increase as powers of 2 up to this number
    params['M'] = []  # no FC layers for U-Net

    assert params['nsides'][0] == NSIDE, "NN nsides values don't match NSIDE! Check the NN architecture!"

    params['K'] = 5  # Polynomial order. (1D array for Deepsphere)

    params["is_resnet"] = True  # conv layer -> resnet block
    params['batch_norm'] = 1  # no normalisation (0) / batch normalization (1) / instance normalisation (2)

    # Estimation of the different template variant fractions
    params["estimate_var_fracs"] = True  # Estimate fractions of template variants
    params["glob_M"] = [256, 128]  # channels for the FCs for the global regression
    params["glob_loss_lambda"] = 1.0  # factor to multiply the L1 loss for the template variants with

    # Regularization.
    # NOTE: THIS ONLY HAS AN EFFECT IF EPISTEMIC UNCERTAINTY ESTIMATION VIA CONCRETE DROPOUT IS DISABLED!
    params['regularization'] = 0.0  # Amount of L2 regularization over the weights (will be divided by the number of weights).

    # Training.
    params['num_steps'] = 25000  # Number of steps to do (total number of maps shown is num_steps * batch_size)
    params['batch_size'] = 64 if HPC else 16  # Number of samples per training batch. Should be a power of 2 for greater speed.
    params['batch_size_val'] = 64 if HPC else 16  # Number of samples for validation
    params['prefetch_batch_buffer'] = 4  # Number of batches to prefetch
    params['prefetch_batch_buffer_val'] = 4  # Number of batches to prefetch
    params['eval_frequency'] = 1  # Frequency of model evaluations during training (influences training time).
    params['scheduler'] = lambda step: 5e-4 * tf.exp(-0.00025 * tf.cast(step, dtype=tf.float32))  # learning rate lambda fct. (e.g. 5e-4),
    # NOTE: function of GLOBAL step (continues when resuming training)!
    # To reset lr when resuming, correct by setting step <- step - step_0
    params['optimizer'] = lambda lr: tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # Determine pixels for each nsides
    # Choose mask slightly larger!
    ROI = hp.reorder(make_mask_total(nside=params["nsides"][-1], mask_ring=True, inner=0, outer=OUTER_RAD + RAD_SAFETY),
                     r2n=True)
    params["indexes"] = get_pixels(ROI, params["nsides"])

    assert not params["aleatoric"] and not params["epistemic"], "Uncertainty estimation for U-Net is not implemented yet!"
    assert not "unmask" in params.keys(), "The key 'unmask' is not available for U-Net!"
    assert not params["project_2d"], "2D CNN is not implemented for U-Net architecture!"

    # Return parameters
    return params
