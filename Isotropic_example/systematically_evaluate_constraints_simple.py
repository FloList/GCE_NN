"""
Systematically evaluate constraints on the Poissonian flux fraction as a function of the PS brightness.
Part 2:
Evaluate the simple estimator, which predicts the Poissonian FFs directly from the SCD predictions.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
import os
import copy
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'rocket'
# ########################################################
print("\n\nFound TF", tf.__version__, ".")
tf.compat.v1.disable_eager_execution()

# ######################################################################################################################
if __name__ == '__main__':
    # ########################################################
    NN_TYPE = "CNN"  # "CNN" or "U-Net"
    GADI = True  # run on Gadi?
    DEBUG = False  # debug mode (verbose and with plots)?
    PRE_GEN = True  # use pre-generated data (CNN only)
    TASK = "TEST"  # "TRAIN" or "TEST"
    RESUME = False  # resume training? WARNING: if False, content of summary and checkpoint folders will be deleted!
    # Options for testing
    TEST_CHECKPOINT = None  # string with global time step to restore. if None: restore latest checkpoint

    NO_PSF = True  # deactivate PSF?
    DO_FAINT = True  # take the NN trained on fainter maps (NO_PSF = True only)?
    DO_FAINT_EVALUATE_ON_BRIGHT = False  # if True: take the NN trained on faint SCDs, but still use the brighter priors for constraining the Poisson flux

    if NO_PSF:
        if DO_FAINT:
            TEST_EXP_PATH = "/scratch/u95/fl9575/GCE_v2/checkpoints/" \
                            "Iso_maps_combined_add_two_faint_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
        else:
            TEST_EXP_PATH = "/scratch/u95/fl9575/GCE_v2/checkpoints/" \
                            "Iso_maps_combined_add_two_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    else:
        if DO_FAINT:
            raise NotImplementedError
        TEST_EXP_PATH = "/scratch/u95/fl9575/GCE_v2/checkpoints/Iso_maps_combined_add_two_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)

    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    checkpoint_path = TEST_EXP_PATH
    # ########################################################

    # Set random seeds for reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    if GADI:
        try:
            JOB_ID = sys.argv[1]
        except IndexError:
            print("NO JOB_ID provided! Setting JOB_ID = 0!")
            JOB_ID = 0
    else:
        JOB_ID = 0
    print("JOB ID is", JOB_ID, ".\n")

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

    data_folder = os.path.join(model.get_path("checkpoints"), "Mixed_PS_Poisson")
    data_folder_orig = copy.copy(data_folder)

    # Faint model has been loaded. Now, use data folder of the BRIGHT priors!
    if DO_FAINT and DO_FAINT_EVALUATE_ON_BRIGHT:
        data_folder = data_folder.replace("_faint", "")

    pred_out_folder = os.path.join(data_folder_orig, "Predictions")
    if DO_FAINT and DO_FAINT_EVALUATE_ON_BRIGHT:
        pred_out_folder += "_bright_data"

    mkdir_p(pred_out_folder)
    save_path = os.path.join(data_folder_orig, "Simple")
    mkdir_p(save_path)
    pred_out_file = "Pred"

    all_taus = np.linspace(0.05, 0.95, 19)
    n_taus = len(all_taus)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    # Check if data exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError("Run 'generate_iso_maps_mixed_PS_Poisson.py' first!")
    else:
        folder_content = os.listdir(data_folder)
        map_files = [m for m in folder_content if ".npy" in m]
        if not len(os.listdir(pred_out_folder)) == len(map_files):
            raise FileNotFoundError("Run 'constrain_Poisson_flux_iso.py' first!")

        # Load
        NN_pred_mix_all = []
        gce_hist = []
        gce_poiss_ff = []
        for i_m, m in enumerate(map_files):
            NN_pred_data = np.load(os.path.join(pred_out_folder, pred_out_file + "_" + m[:-4]) + ".npz",
                                   allow_pickle=True)
            if len(NN_pred_mix_all) == 0:
                NN_pred_mix_all = NN_pred_data["NN_pred_mix_all"]
                gce_hist = NN_pred_data["gce_hist"]
                gce_poiss_ff = NN_pred_data["gce_poiss_ff"]
            else:
                NN_pred_mix_all = np.concatenate([NN_pred_mix_all, NN_pred_data["NN_pred_mix_all"]], axis=1)
                gce_hist = np.concatenate([gce_hist, NN_pred_data["gce_hist"]], axis=0)
                gce_poiss_ff = np.concatenate([gce_poiss_ff, NN_pred_data["gce_poiss_ff"]], axis=0)
        print("Predictions for mixed PS/Poisson data LOADED!")

    # ROI parameters
    mean_exp = generator_test.settings_dict["exp"].mean()

    # Get cumulative predictions for mixed maps
    n_eval = NN_pred_mix_all.shape[1]
    NN_pred_mix_all_cum = NN_pred_mix_all.cumsum(2)

    # We will mostly work with the median predictions for the mixed / PS data here
    median_ind = (n_taus - 1) // 2
    mix_hist = NN_pred_mix_all[median_ind]
    mix_hist_cum = NN_pred_mix_all_cum[median_ind]

    # Define the check_vals such that we get a calibrated estimator (see constrain_Poisson_flux_iso.py)
    n_cal = 4 * n_eval // 5
    n_val = n_eval // 5
    Poisson_frac_constraint = np.zeros((n_taus, n_cal))

    if NO_PSF:
        if DO_FAINT:
            check_vals = np.asarray([-0.0225, -0.2756824, -0.42388694, -0.53514489, -0.67422501,
                                     -0.73154853, -0.75780217, -0.79082796, -0.83611944, -0.89851441,
                                     -0.96824209, -1.05573898, -1.29483705, -1.34758148, -1.40868692,
                                     -1.64143633, -1.70857328, -1.76896754, -1.83800109])  # the first value is very slightly off (0.25%), difficult to tune
        else:
            check_vals = np.asarray([0.33976004, 0.03212888, -0.13022727, -0.23530114, -0.32122574,
                                     -0.38773713, -0.44045476, -0.50049715, -0.55076788, -0.58149048,
                                     -0.61769033, -0.66304489, -0.71456483, -0.73553192, -0.77009073,
                                     -0.82450787, -0.88992826, -0.94257594, -1.01752257])
    else:
        check_vals = np.asarray([0.14752859, -0.16820803, -0.26806663, -0.33695885, -0.38020523,
                                 -0.40960847, -0.43735173, -0.4669681, -0.49884161, -0.53468994,
                                 -0.56189077, -0.59661545, -0.64684144, -0.71505463, -0.76586588,
                                 -0.83825055, -0.90777006, -0.96435328, -1.04023497])

    # check_vals are defined from highest to lowest -> flip
    check_vals = check_vals[::-1]

    # Load the delta dN/dlogF histograms
    sys_data_path = os.path.join(checkpoint_path, "Mixed_PS_Poisson", "Systematic")
    sys_data_pred_path = os.path.join(sys_data_path, "Predictions")
    sys_data_pred_file = os.path.join(sys_data_pred_path, "Pred.npy")

    try:
        sys_data = np.load(sys_data_pred_file)
    except FileNotFoundError:
        raise FileNotFoundError("Run 'systematically_evaluate_constraints_part_1.py' first!")

    # Get median prediction and flatten
    _, n_Poiss_frac, n_counts_per_PS, n_realisations, _ = sys_data.shape
    sys_data_median = sys_data[median_ind, :, :, :, :]  # Median tau
    sys_data_median_flat = np.reshape(sys_data_median, [-1, len(bin_centres)])
    sys_data_median_flat_cum = sys_data_median_flat.cumsum(1)

    # Evaluate for different confidence levels alpha (here: tau, somewhat confusing)
    n_eval_sys = sys_data_median_flat.shape[0]
    sys_preds_flat = np.zeros((n_taus, n_eval_sys))

    for i_tau, tau in enumerate(all_taus):
        sys_data_loc = sys_data_median_flat

        bin_ind_above = np.argmax(bin_centres > check_vals[i_tau])
        bin_ind_below = bin_ind_above - 1
        alpha = (check_vals[i_tau] - bin_centres[bin_ind_below]) \
                / (bin_centres[bin_ind_above] - bin_centres[bin_ind_below])

        for i in range(n_eval_sys):
            constraint = alpha * sys_data_median_flat_cum[i, bin_ind_above] \
                         + (1 - alpha) * sys_data_median_flat_cum[i, bin_ind_below]
            sys_preds_flat[i_tau, i] = constraint

    # Bring to shape n_taus x n_Poiss_frac x n_counts_per_PS x n_realisations
    sys_preds = np.transpose(np.reshape(sys_preds_flat.T, [n_Poiss_frac, n_counts_per_PS, n_realisations, n_taus]), [3, 0, 1, 2])

    # Define counts per PS and Poisson FF arrays
    counts_per_PS_ary = np.logspace(-1, 3, 11)[:7]
    Poiss_fracs = np.linspace(0.0, 1.0, 6)

    # Define the colour map
    x_vec = np.asarray([0, 0.5, 1])
    cmap_orig = copy.copy(cc.cm.CET_D3_r)
    cmap_3_vals = cmap_orig(x_vec)

    cmap_gkr_orig = copy.copy(cc.cm.diverging_gkr_60_10_c40_r)
    cmap_gkr_3_vals = cmap_gkr_orig(x_vec)
    cmap_new_3_vals = np.vstack([cmap_3_vals[0], cmap_gkr_3_vals[1], cmap_3_vals[2]])
    N_interp = 256
    cmap_new_ary = np.asarray([np.interp(np.linspace(0, 1, N_interp), x_vec, cmap_new_3_vals[:, i])
                               for i in range(4)]).T
    cmap_new = mpl.colors.ListedColormap(cmap_new_ary)

    #colors_constraint = cc.cm.diverging_gkr_60_10_c40_r(np.linspace(0, 1, n_Poiss_frac))
    colors_constraint = cmap_new(np.linspace(0, 1, n_Poiss_frac))
    counts_per_PS_ary_max_ind = 7

    # Make a plot
    if DO_FAINT and DO_FAINT_EVALUATE_ON_BRIGHT:
        out_file_plot = "systematic_constraints_simple_small_bright_data.pdf"
    else:
        out_file_plot = "systematic_constraints_simple_small.pdf"
    plot_ind_tau = 18  # -1: 95% confidence
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.77))
    ax.set_xscale("log")
    x_values = counts_per_PS_ary[:counts_per_PS_ary_max_ind]
    lw = 1.5
    for i_Poiss_frac, Poiss_frac in enumerate(Poiss_fracs):
        median_constraint = np.median(sys_preds[plot_ind_tau, i_Poiss_frac, :counts_per_PS_ary_max_ind, :], 1)  # median over the maps
        scatter_low = median_constraint \
                      - np.quantile(sys_preds[plot_ind_tau, i_Poiss_frac, :counts_per_PS_ary_max_ind, :], 0.16, 1)
        scatter_high = np.quantile(sys_preds[plot_ind_tau, i_Poiss_frac, :counts_per_PS_ary_max_ind, :], 0.84, 1) \
                       - median_constraint
        yerr = np.vstack([scatter_low, scatter_high])
        ax.errorbar(x=x_values, y=median_constraint, yerr=yerr, lw=lw, color=colors_constraint[i_Poiss_frac], capsize=3,
                    marker="o", ms=4, markeredgewidth=1, elinewidth=lw, zorder=2)
        ax.axhline(Poiss_frac, color=colors_constraint[i_Poiss_frac], ls="--", lw=1, zorder=1)
        # if i_Poiss_frac == 0:
            # ax.text(0.1, Poiss_frac + 0.025, r"True $\eta_P$", color=colors_constraint[i_Poiss_frac], size="small")
    ax.set_xlabel("Expected counts per PS")
    ax.set_ylabel(r"Poisson flux fraction $\eta_P$")
    plt.tight_layout()
    fig.savefig(os.path.join(sys_data_path, out_file_plot), bbox_inches="tight")


# Helper function to set the size of an axis
# def set_size(w,h, ax=None):
#     """ w, h: width, height in inches """
#     if not ax: ax=plt.gca()
#     l = ax.figure.subplotpars.left
#     r = ax.figure.subplotpars.right
#     t = ax.figure.subplotpars.top
#     b = ax.figure.subplotpars.bottom
#     figw = float(w)/(r-l)
#     figh = float(h)/(t-b)
#     ax.figure.set_size_inches(figw, figh)
#
# set_size(h=3.843, w=3.729, ax=ax)
