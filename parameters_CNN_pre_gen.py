"""
The function in this file returns the parameter for the GCE CNN with pre-generated data
(see arXiv:2006.12504).
"""
import healpy as hp
import tensorflow as tf
from gce_utils import get_pixels
from NPTFit.create_mask import make_mask_total


def get_params_CNN_pre_gen(HPC):
    params = dict()

    # General settings
    NAME = "GCE_and_background"  # "GCE_and_diffuse", "GCE_all", "GCE_smooth_and_PS"
    PROJECT_2D = False  # If True: don't use a graph CNN but project data to 2D image and use standard 2D CNN
                       # NOTE: experimental, many options (e.g. uncertainty estimation) are not implemented!
    OUTER_RAD = 25  # outer ROI radius to use
    RAD_SAFETY = 2 if PROJECT_2D else 1  # require that all the pixels within OUTER_RAD + RAD_SAFETY are within the ROI
    MODELS = ["dif_O_pibs", "dif_O_ic", "iso", "bub", "gce", "gce_PS"]  # Templates
    MODEL_NAMES = [r"diffuse $\pi^0$ + BS", "diffuse IC", "isotropic", r"$\it{Fermi}$ bubbles", "GCE DM", "GCE PS"]  # Template names
    EXP_NAME = NAME + "_pre_gen"  # name of experiment
    data_folder = '/scratch/u95/fl9575/GCE_v2/data/' + NAME if HPC else '/home/flo/PycharmProjects/GCE/data/' + NAME

    # GCE settings
    params["models"] = MODELS
    params["model_names"] = MODEL_NAMES
    params["project_2d"] = PROJECT_2D
    params["outer_rad"] = OUTER_RAD
    params["rad_safety"] = RAD_SAFETY
    params["n_params"] = len(MODELS)  # length of labels = number of models
    params["data_folder"] = data_folder
    params["unmask"] = False  # unmask = True -> run on ENTIRE sphere instead of fraction! MUCH SLOWER!
    params["ring2nest"] = True  # if input data is stored in RING format
    params["ROI_fermi"] = OUTER_RAD  # angle around the GC that contains data: Fermi data outside will be masked
    params["mask_3FGL_fermi"] = False  # in the Fermi counts map: mask the 3FGL sources?
    params["rel_counts"] = True  # scale the pixel values by the total number of counts in the map?
    params["append_tot_counts"] = True  # if params["rel_counts"] = True: append total number of counts to input for first FC layer?
    params["aleatoric"] = True if not PROJECT_2D else False  # if True: estimate aleatoric uncertainties (including covariance matrix)
    params["alea_only_var"] = True  # if True: only estimate variances, no correlations
    params["alea_split"] = False  # if False: directly use log likelihood loss and fit means and covariances simultaneously
                                      # if True: step 1: train means only, using l2 loss.
                                      #          step 2: after training the means, fix the weights and only fit covariances
    params["alea_step"] = 1  # if splitting: step 1 or 2
    params["covar_scaling"] = 1.0  # scale aleatoric uncertainty covariance matrix for numerical stability
    params["epistemic"] = True if not PROJECT_2D else False  # estimate epistemic uncertainty
    params["epi_n_dropout"] = 10  # how many samples to evaluate to use for estimating epistemic uncertainty during training
    params["epi_p_init_min"] = 0.01  # min. initial dropout prob.
    params["epi_p_init_max"] = 0.01  # max. initial dropout prob.
    params["epi_prior_length_scale"] = 1.0  # epistemic prior length scale l (e.g. 1e-4)
    params["epi_n_training_samples"] = 6e5 if HPC else 2.2e4  # number of training samples (needed for Concrete Dropout priors)
    params["epi_dropout_output_layer"] = True  # do MC dropout for last FC layer?

    # Training data selection settings
    params["cond_on_training_data"] = None  # lambda function of a single map
    params["cond_on_training_labels"] = None  # lambda function of a flux contribution vector
    params["prob_for_conditions"] = 0.0  # impose these conditions with this probability
                                         # (the resulting proportion of samples fullfilling this condition will be higher!)

    # Template settings
    # NOTE: the following options are NOT implemented for PROJECT_2D!
    # NOTE: templates are assumed to be in RING ordering and will be converted to NESTED
    # No need to use template iso since it is homogeneous in space, so doesn't provide any information
    params["template_path"] = "/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data" if HPC else "/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data"
    params["template_names"] = None
    params["exp_name"] = "fermidata_exposure"  # name of exposure file
    params["template_blocks"] = False  # if True: split up convolutional part of the NN in different blocks for different templates
                                      # otherwise: just append map * template as additional channels for the first convolutional layer
    params["append_map"] = False  # if True: append the full map (unconvolved) to the output of the conv. layers before FC layers

    # Set parameters for DeepSphere
    params['dir_name'] = EXP_NAME

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['new_weighting'] = True  # use new weighting scheme, with kernel width taken from https://openreview.net/pdf?id=B1e3OlStPB
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['last_act'] = 'softmax'  # Last activation: softmax (sum(labels) = 1, WORKS MUCH BETTER!) or sigmoid (enforce sum(fractions) = 1 using lambda)
    params['statistics'] = None  # Set to None
    params['loss'] = 'l2'  # l2 loss for regression (also includes normal max. likelihood loss for aleatoric == True) or
                           # l1 loss for regression (also include Laplace max. likelihood loss for aleatoric == True)

    # Data transformation (for tf tensors, not np!)
    # NOTE: The data transformation is applied AFTER removing the exposure correction present in simulated & Fermi data
    if params["rel_counts"]:
        params["data_trafo"] = lambda data: data
    else:
        params["data_trafo"] = lambda data: data

    # Architecture.
    params['nsides'] = [128, 64, 32, 16, 8, 4, 2, 1]  # number of nsides per layer (for PROJECT_2D: only the first value will be used)
    params['F'] = [32, 64, 128, 256, 256, 256, 256]  # (Graph) convolutional layers: number of feature maps.
    # params['F'] = [16, 32, 64, 128, 128, 128, 128]
    params['M'] = [2048, 512, params["n_params"]]  # Fully connected layers: output dimensionalities.
    # params['M'] = [1024, 256, params["n_params"]]

    if PROJECT_2D:
        params['input_shape'] = [128, 128]
        params['K'] = [[5, 5]] * len(params['F'])  # Kernel sizes (list of 2D arrays for 2DCNN)
        params['p'] = [1] * len(params['F'])  # Strides for cnn2D
        params['pool_fac'] = [2] * len(params['F'])  # pooling factors for cnn2D: use EITHER stride OR pooling!`
    else:
        params['K'] = [5] * len(params['F'])  # Polynomial orders. (1D array for Deepsphere)
        params["glob_avg"] = False  # Global averaging layer (avg. over spatial dims.) before FC layers (makes only sense if not convolving down to 1 px)

    params["is_resnet"] = [False] * (len(params['F']) - 0)  # conv layer -> resnet block
    params['batch_norm'] = [1] * len(params['F'])  # Batch normalization (1) / instance normalisation (2)

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
    params['scheduler'] = lambda step: 5e-4 * tf.exp(-0.00025 * tf.cast(step, dtype=tf.float32))  # learning rate lamda fct. (e.g. 5e-4),
                         # NOTE: function of GLOBAL step (continues when resuming training)!
                         # To reset lr when resuming, correct by setting step <- step - step_0
    params['optimizer'] = lambda lr: tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # If PROJECT_2D: always need to unmask!
    if PROJECT_2D:
        params["unmask"] = True

    # Determine pixels for each nsides
    # Choose mask slightly larger!
    ROI = hp.reorder(make_mask_total(nside=params["nsides"][-1], mask_ring=True, inner=0, outer=OUTER_RAD + RAD_SAFETY),
                     r2n=True)
    params["indexes"] = get_pixels(ROI, params["nsides"])

    # Flags for pre-generated data
    params["estimate_var_fracs"] = False
    params["model_vars"] = [None] * len(MODELS)

    # Return parameters
    return params
