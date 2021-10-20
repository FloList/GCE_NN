"""
Systematically evaluate constraints on the Poissonian flux fraction as a function of the PS brightness.
Part I:
a) Generate maps and save
b) Get NN predictions for histograms and save
"""

import shutil
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from scipy import stats
import sys
from gce_utils import *
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'rocket'

######################################################
HPC = get_is_HPC()
######################################################
if not HPC:
    new_lib = '/usr/local/lib'
    try:
        if not new_lib in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] += ':'+new_lib
    except:
        os.environ['LD_LIBRARY_PATH'] = new_lib
    os.chdir('/home/flo/PycharmProjects/GCE/DeepSphere/Fast_PS_generation')
    sys.path.append('/home/flo/PycharmProjects/GCE/DeepSphere/Fast_PS_generation')
from NPTFit import create_mask as cm  # Module for creating masks
from ps_mc_fast import PDFSampler, run
######################################################

# Settings
nside = 256
outer_ring = 25.0  # choose EXACTLY the same mask and exposure that will be used for the NN training!
inner_band = 0.0
MASK_TYPE = "None"  # mask for known bright PSs: either: "None", "3FGL", "4FGL"
NO_PSF = True
DO_FAINT = True
npix = hp.nside2npix(nside)
fermi_folder = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data_573w' if HPC \
          else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data_573w'
fermi_folder += "/fermi_data_" + str(nside)
#############
# CHOOSE MODEL
T_P = "iso"
T_NP = T_P + "_PS"
#############

# Settings for saving / loading
if NO_PSF:
    if DO_FAINT:
        checkpoint_path = '/scratch/u95/fl9575/GCE_v2/checkpoints/' \
                          'Iso_maps_combined_add_two_faint_no_PSF_IN_bs_256_softplus_pre_gen'
    else:
        checkpoint_path = '/scratch/u95/fl9575/GCE_v2/checkpoints/' \
                          'Iso_maps_combined_add_two_no_PSF_IN_bs_256_softplus_pre_gen'
else:
    if DO_FAINT:
        raise NotImplementedError
    checkpoint_path = '/scratch/u95/fl9575/GCE_v2/checkpoints/Iso_maps_combined_add_two_IN_bs_256_softplus_pre_gen'

save_path = os.path.join(checkpoint_path, "Mixed_PS_Poisson", "Systematic")
mkdir_p(save_path)
filename_out = os.path.join(save_path, "Delta_dNdF_data")

if len(os.listdir(save_path)) == 0:
    print("Starting data generation...")
    # Exposure map
    fermi_exp = get_template(fermi_folder, "exp")
    fermi_exp = hp.reorder(fermi_exp, r2n=True)
    mean_exp_fermi = np.mean(fermi_exp)
    fermi_rescale = fermi_exp / mean_exp_fermi
    exp = np.ones_like(fermi_exp)
    mean_exp = exp.mean()

    prior_dict = dict()

    # PSF: use Fermi-LAT PSF?
    if NO_PSF:
        pdf = None
    else:
        pdf = get_Fermi_PDF_sampler()

    # Set up the mask for the ROI
    total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band,
                                        mask_ring=True, inner=0, outer=outer_ring, nside=nside)

    if MASK_TYPE == "3FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_folder, "3FGL_mask"))).astype(bool)
    elif MASK_TYPE == "4FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_folder, "4FGL_mask"))).astype(bool)

    total_mask_neg = hp.reorder(total_mask_neg, r2n=True)
    n_pix_ROI = (1 - total_mask_neg).sum()

    # Make masked template map
    T = get_template(fermi_folder, T_P)
    T = hp.reorder(T, r2n=True)
    T_counts = T_flux = T / fermi_rescale

    T_counts_masked = T_counts * (1 - total_mask_neg)
    T_flux_masked = T_flux * (1 - total_mask_neg)

    inds_compress_mask = (1 - total_mask_neg).astype(bool)
    T_counts_masked_compressed = T_counts_masked[inds_compress_mask]

    # Number of realisations for each PS brightness
    n_realisations = 64
    counts_per_PS_ary = np.logspace(-1, 3, 11)[:7]
    Poiss_fracs = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_Poiss_fracs = len(Poiss_fracs)
    n_counts_per_PS = len(counts_per_PS_ary)
    map_ary = np.zeros((n_Poiss_fracs, n_counts_per_PS, n_realisations, n_pix_ROI, 2))  # shape: Poisson fraction x counts per PS x realisations x pixels x 2 (Poiss, PS)

    n_counts_exp = 50000
    A_P = n_counts_exp / n_pix_ROI
    deactivate_PS_draw = False

    # Prepare NP maps
    T_flux_masked_norm = T_flux_masked / T_flux_masked.sum()
    # Template needs to be normalised to sum up to unity for the new implementation!
    # Might need to do this twice because of rounding errors
    while T_flux_masked_norm.sum() > 1.0:
        T_flux_masked_norm /= T_flux_masked_norm.sum()

    # Iterate over the Poisson fractions
    for i_Poiss_frac, Poiss_frac in enumerate(Poiss_fracs):
        print("Poisson fraction:", Poiss_frac)

        # Iterate over the counts per PS
        for i_counts, counts_per_PS in enumerate(counts_per_PS_ary):
            print("  Counts per PS:", counts_per_PS)

            # P maps
            if Poiss_frac > 0:
                maps_P = np.asarray([np.random.poisson(Poiss_frac * A_P * T_counts_masked_compressed)
                                     for _ in range(n_realisations)])
            else:
                maps_P = np.zeros((n_realisations, n_pix_ROI))
            map_ary[i_Poiss_frac, i_counts, :, :, 0] = maps_P

            # Delta fct. dN/dF:
            tot_exp_counts = (1 - Poiss_frac) * n_counts_exp
            tot_exp_PS = tot_exp_counts / counts_per_PS
            exp_flux_per_PS = counts_per_PS / mean_exp
            if deactivate_PS_draw:
                N_PS = int(np.round(tot_exp_PS))  # fixed number of sources
                flux_arr = exp_flux_per_PS * np.ones(N_PS)  # define flux array
                flux_arr = np.tile(flux_arr, [n_realisations, 1])
            else:
                N_PS = np.random.poisson(tot_exp_PS, size=n_realisations)  # draw the number of sources from a Poisson distr.
                flux_arr = [exp_flux_per_PS * np.ones(N_PS[i]) for i in range(n_realisations)]

            # Generate
            for i_r in range(n_realisations):
                map_NP = run(flux_arr[i_r], T_flux_masked_norm, exp, pdf, "", save=False, getnopsf=False, getcts=False,
                                         upscale_nside=16384, verbose=False, is_nest=True)
                map_ary[i_Poiss_frac, i_counts, i_r, :, 1] = map_NP[inds_compress_mask]

    # Print stats
    mean_Poiss_fracs = map_ary.sum(3).mean(2).mean(1)[:, 0] \
                       / (map_ary.sum(3).mean(2).mean(1)[:, 0] + map_ary.sum(3).mean(2).mean(1)[:, 1])
    print("Mean Poisson fraction:", mean_Poiss_fracs)
    tot_counts_avg = map_ary.sum(4).sum(3).mean(2).mean(1).mean(0)
    print("Mean tot. counts:", tot_counts_avg)

    # Save
    np.save(filename_out, map_ary)
    print("Maps SAVED!")

else:
    # Load data
    map_ary = np.load(filename_out + ".npy")

    import tensorflow as tf
    from deepsphere_GCE_workflow import build_NN

    print("\n\nFound TF", tf.__version__, ".")
    tf.compat.v1.disable_eager_execution()

    NN_TYPE = "CNN"  # "CNN" or "U-Net"
    GADI = True  # run on Gadi?
    DEBUG = False  # debug mode (verbose and with plots)?
    PRE_GEN = True  # use pre-generated data (CNN only)
    TASK = "TEST"  # "TRAIN" or "TEST"
    RESUME = False  # resume training? WARNING: if False, content of summary and checkpoint folders will be deleted!
    # Options for testing
    TEST_CHECKPOINT = None  # string with global time step to restore. if None: restore latest checkpoint

    if NO_PSF:
        if DO_FAINT:
            TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_add_two_faint_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
        else:
            TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_add_two_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    else:
        if DO_FAINT:
            raise NotImplementedError
        TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_add_two_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)

    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    # Set random seeds for reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # Plot settings
    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    bin_edges = copy.copy(params["gce_hist_bins"])

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    save_path_pred = os.path.join(save_path, "Predictions")
    mkdir_p(save_path_pred)
    pred_out_file = os.path.join(save_path_pred, "Pred")

    all_taus = np.linspace(0.05, 0.95, 19)
    n_taus = len(all_taus)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    if not os.path.exists(os.path.join(save_path_pred, pred_out_file + ".npy")):
        # Predict
        map_ary_tot = map_ary.sum(4)  # add Poisson + PS
        map_ary_tot_flat = np.reshape(map_ary_tot, [-1, map_ary_tot.shape[3]])
        n_eval = map_ary_tot_flat.shape[0]
        NN_pred_all_flat = np.zeros((n_eval, len(bin_centres), n_taus))
        for i_tau, tau in enumerate(all_taus):
            NN_pred = model.predict({"data": map_ary_tot_flat[:n_eval, :]}, None, False,
                                    tau_hist=tau * np.ones((n_eval, 1)))
            NN_pred_all_flat[:, :, i_tau] = NN_pred["gce_hist"][:, :, 0]

        NN_pred_all = np.reshape(NN_pred_all_flat, list(map_ary_tot.shape[:3]) + [len(bin_centres), n_taus])
        NN_pred_all = np.transpose(NN_pred_all, [4, 0, 1, 2, 3])  # taus x Poisson frac x PS brightness x realisations x bins
        np.save(pred_out_file, NN_pred_all)
        print("Predictions for delta dN/dF data SAVED!")
        sys.exit(0)
    else:
        print("Predictions for delta dN/dF data exist already! Exiting...")
        #NN_pred_all = np.load(os.path.join(save_path_pred, pred_out_file + ".npy"))
        sys.exit(0)
