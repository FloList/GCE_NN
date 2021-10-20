"""
This script computes constraints on the Poissonian flux fraction using the SIMPLE (less powerful) estimator
discussed in arXiv:2107.09070.
"""
import shutil
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
        maps_loc = [m for m in folder_content if "Maps_" + str(JOB_ID) in m]
        maps_loc.sort()
        n_maps_loc = len(maps_loc)

        # Check if NN predictions have been computed
        if not len(os.listdir(pred_out_folder)) == len(map_files):
            # Predict
            for i_m, m in enumerate(maps_loc):
                data_loc = np.load(os.path.join(data_folder, m), allow_pickle=True)[()]
                n_eval = data_loc["label"].shape[0]
                NN_pred_mix_all = np.zeros((n_taus, n_eval, len(bin_centres)))
                for i_tau, tau in enumerate(all_taus):
                    NN_pred_mix = model.predict({"data": data_loc["data"][:n_eval, :]}, None, False,
                                                tau_hist=tau * np.ones((n_eval, 1)))
                    NN_pred_mix_all[i_tau] = NN_pred_mix["gce_hist"][:, :, 0]

                np.savez(os.path.join(pred_out_folder, pred_out_file + "_" + m[:-4]),
                         NN_pred_mix_all=NN_pred_mix_all, label=data_loc["label"], gce_hist=data_loc["gce_hist"],
                         gce_poiss_ff=data_loc["gce_poiss_ff"])
            print("Predictions for mixed PS/Poisson data SAVED!")
            sys.exit(0)
        else:
            # Load
            NN_pred_mix_all = []
            gce_hist = []
            gce_poiss_ff = []
            for i_m, m in enumerate(map_files):
                NN_pred_data = np.load(os.path.join(pred_out_folder, pred_out_file + "_" + m[:-4])
                                       + ".npz", allow_pickle=True)
                if len(NN_pred_mix_all) == 0:
                    NN_pred_mix_all = NN_pred_data["NN_pred_mix_all"]
                    gce_hist = NN_pred_data["gce_hist"]
                    gce_poiss_ff = NN_pred_data["gce_poiss_ff"]
                else:
                    NN_pred_mix_all = np.concatenate([NN_pred_mix_all, NN_pred_data["NN_pred_mix_all"]], axis=1)
                    gce_hist = np.concatenate([gce_hist, NN_pred_data["gce_hist"]], axis=0)
                    gce_poiss_ff = np.concatenate([gce_poiss_ff, NN_pred_data["gce_poiss_ff"]], axis=0)
            print("Predictions for mixed PS/Poisson data LOADED!")

    # Plot the distribution of the Poissonian fraction.
    # If the maps were summed "naively" at random, this should give a triangular distribution,
    # if sorted in reverse order a uniform distribution
    plt.figure()
    Poisson_frac = np.squeeze(gce_poiss_ff, -1)  # second dimension can be squeezed
    plt.hist(Poisson_frac, bins=np.linspace(0, 1, 11), density=True)

    # ROI parameters
    mean_exp = generator_test.settings_dict["exp"].mean()

    # Get cumulative predictions for mixed maps
    n_eval = NN_pred_mix_all.shape[1]
    NN_pred_mix_all_cum = NN_pred_mix_all.cumsum(2)

    # We will mostly work with the median predictions for the mixed / PS data here
    median_ind = (n_taus - 1) // 2
    mix_hist = NN_pred_mix_all[median_ind]
    mix_hist_cum = NN_pred_mix_all_cum[median_ind]

    # 2. Attempt: very simple: look at mixed CDF prediction in bin where Poissonian CDF reaches (almost) 1
    poiss_threshold = 0.99
    # i_check = np.argmin(np.median(NN_pred_mix_all_cum[median_ind_Poiss, :, :], 0) < poiss_threshold)
    # for i in range(256):
    #     print("FF Poiss.:", Poisson_frac[i], "Mixed CDF value:", mix_hist_cum[i, i_check])
    #     # print("PS CDF value:", PS_hist_cum[i, i_check])

    # Is this criterion "well-calibrated"? NO...
    FF_lies_below_value = np.zeros((n_taus, n_eval))
    Poisson_frac_constraint = np.zeros((n_taus, n_eval))
    interp_bins = np.linspace(bin_centres[0], bin_centres[-1], 10001)
    check_vals = np.zeros(n_taus)

    for i_tau, tau in enumerate(all_taus):
        interpol_CDF = np.interp(interp_bins, bin_centres, np.median(NN_pred_mix_all_cum[i_tau, :, :], 0))
        i_check_interpol = np.argmin(interpol_CDF < poiss_threshold)
        check_val = interp_bins[i_check_interpol]
        check_vals[i_tau] = check_val
        bin_ind_above = np.argmax(bin_centres > check_val)
        bin_ind_below = bin_ind_above - 1
        print(tau, check_val)
        alpha = (check_val - bin_centres[bin_ind_below]) / (bin_centres[bin_ind_above] - bin_centres[bin_ind_below])

        for i in range(n_eval):
            # i_check = np.argmin(pred_hist_cum_all[i_tau, i, :] < poiss_threshold)
            Poisson_frac_constraint[i_tau, i] = alpha * mix_hist_cum[i, bin_ind_above] \
                                                + (1 - alpha) * mix_hist_cum[i, bin_ind_below]
            if Poisson_frac[i] <= Poisson_frac_constraint[i_tau, i]:
                FF_lies_below_value[i_tau, i] = 1
    print(FF_lies_below_value.mean(1))
    print(all_taus[::-1])
    fig_cal, ax_cal = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax_cal.plot(all_taus[::-1], FF_lies_below_value.mean(1), ls="none", marker="o", mfc="white", mec="k", ms=5)
    ax_cal.plot(all_taus[::-1], all_taus[::-1], "k-", lw=1)

    # Do the same, but DEFINE the check_vals such that we get a calibrated estimator
    # only calibrate on a fraction of the data and evaluate on the rest
    n_cal = 4 * n_eval // 5
    n_val = n_eval // 5
    Poisson_frac_constraint = np.zeros((n_taus, n_cal))
    interp_bins = np.linspace(bin_centres[0], bin_centres[-1], 10001)
    FF_lies_below_value = np.zeros((n_taus, n_cal))

    tol = 0.001
    dt = 0.1
    counter = 1
    max_iter = 1000
    i_tau_c = median_ind
    n_print = 10

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

    # while np.any(np.abs(FF_lies_below_value.mean(1) - all_taus[::-1]) > tol) and counter <= max_iter:
    #     FF_lies_below_value = np.zeros((n_taus, n_cal))
    #     if counter % n_print == 0:
    #         print("It.", counter)
    #     for i_tau in range(len(all_taus)):
    #
    #         bin_ind_above = np.argmax(bin_centres > check_vals[i_tau])
    #         bin_ind_below = bin_ind_above - 1
    #         alpha = (check_vals[i_tau] - bin_centres[bin_ind_below]) / (bin_centres[bin_ind_above] - bin_centres[bin_ind_below])
    #
    #         for i in range(n_cal):
    #             # i_check = np.argmin(pred_hist_cum_all[i_tau, i, :] < poiss_threshold)
    #             Poisson_frac_constraint[i_tau, i] = alpha * NN_pred_mix_all_cum[i_tau_c, i, bin_ind_above] \
    #                                                 + (1 - alpha) * NN_pred_mix_all_cum[i_tau_c, i, bin_ind_below]
    #             if Poisson_frac[i] <= Poisson_frac_constraint[i_tau, i]:
    #                 FF_lies_below_value[i_tau, i] = 1
    #
    #     update = all_taus[::-1] - FF_lies_below_value.mean(1)
    #     check_vals = check_vals + dt * update
    #     check_vals.sort()
    #     check_vals = check_vals[::-1]
    #     if counter % n_print == 0:
    #         print("  Mean cal. error:", np.abs(update).mean())
    #         print("  Max cal. error:", np.abs(update).max())
    #     counter += 1

    # Calibration plot for the data it was calibrated on
    fig_cal, ax_cal = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax_cal.plot(all_taus[::-1], all_taus[::-1], "k-", lw=2)
    ax_cal.plot(all_taus[::-1], FF_lies_below_value.mean(1), "ko", lw=0, markersize=8, mfc="white", markeredgewidth=2)
    plt.tight_layout()
    fig_cal.savefig(os.path.join(save_path, "calibration_plot_calibrated_data.pdf"), bbox_inches="tight")

    # Now: evaluate on the rest of the data
    constraints = np.zeros((n_taus, n_val))
    for i_tau in range(len(all_taus)):
        bin_ind_above = np.argmax(bin_centres > check_vals[i_tau])
        bin_ind_below = bin_ind_above - 1
        for i in range(n_cal, n_eval):
            alpha = (check_vals[i_tau] - bin_centres[bin_ind_below]) \
                    / (bin_centres[bin_ind_above] - bin_centres[bin_ind_below])
            constraint = alpha * NN_pred_mix_all_cum[median_ind][i, bin_ind_above] \
                         + (1 - alpha) * NN_pred_mix_all_cum[median_ind][i, bin_ind_below]
            constraints[i_tau, i - n_cal] = constraint
            # print("FF Poiss.: {:2.1f}%, Constraint: {:#2.1f}%".format(100 * Poisson_frac[i - n_eval // 2], 100 * constraint))
            # print("PS CDF value:", PS_hist_cum[i, i_check])
        print("tau:{:1.2f}, coverage:{:1.4f}".format(all_taus[::-1][i_tau], (constraints[i_tau] > Poisson_frac[n_cal:n_eval]).mean()))

    # Make a calibration plot for the unseen data
    fig_cal, ax_cal = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax_cal.plot(all_taus[::-1], all_taus[::-1], "k-", lw=2)
    ax_cal.plot(all_taus[::-1], (constraints > Poisson_frac[n_cal:n_eval]).mean(1),
                "ko", lw=0, markersize=8, mfc="white", markeredgewidth=2)
    plt.tight_layout()
    fig_cal.savefig(os.path.join(save_path, "calibration_plot_uncalibrated_data.pdf"), bbox_inches="tight")

    # Plot the check_vals
    plt.figure(figsize=(3.5, 3.5))
    plt.plot(all_taus[::-1], check_vals, "ko", lw=0, markersize=8, mfc="white", markeredgewidth=2)
    plt.xlabel("Confidence level")
    plt.ylabel(r"$\phi$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "phi_vs_alpha.pdf"), bbox_inches="tight")

    # Plot some examples (from the dataset it was not calibrated on)
    tau_inds_eval_constraint = [0, 5, 9]  # NOTE: THE ORDER IS REVERSED, SO THE INDICES CORRESPOND TO 1 - tau!!!
                                          # for example [0, 5, 9] -> 95%, 70%, 50%
    plot_start, plot_end = 0, 30
    plt.ioff()
    plot_range, constraint_offset = range(n_cal + plot_start, n_cal + plot_end), n_cal
    # plot_range, constraint_offset = range(plot_start, plot_end), 0
    width = np.diff(bin_centres).mean()

    for plot_ind in plot_range:
        fig, ax = plt.subplots(1, 1)
        ax.plot(bin_centres, mix_hist_cum[plot_ind], "k-", lw=2)  # NN prediction for mixed PS/Poisson map: median
        if i_tau_c != median_ind:
            ax.plot(bin_centres, NN_pred_mix_all_cum[i_tau_c, plot_ind], "k--", lw=2)  # NN prediction for mixed PS/Poisson map: quantile for which calibration was done
        # Plot cumulative histogram estimates (faint)
        for i_tau in range(len(all_taus)):
            if i_tau < n_taus - 1:
                # Draw the next section of the cumulative histogram in the right colour
                for i in range(len(bin_centres)):
                    # Draw the next section of the cumulative histogram in the right colour
                    ax.fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                    y1=NN_pred_mix_all_cum[i_tau, plot_ind, i],
                                    y2=NN_pred_mix_all_cum[i_tau + 1, plot_ind, i],
                                    color=colors[i_tau], lw=0, alpha=0.5)
        ax.plot(bin_centres, gce_hist[plot_ind, :, 0].cumsum(), color="darkslateblue", ls="-", lw=2)  # True label (PS only, 0: GCE)
        ax.axhline(Poisson_frac[plot_ind], color="k", lw=1, ls="--")
        ax.text(0, 0.05, "True Poisson FF:", ha="left")
        ax.text(2.3, 0.05, "{:2.1f}%".format(100 * Poisson_frac[plot_ind]), ha="right")
        ax.text(0, 0.01, "Constraints:", ha="left")
        for i_spacing, i_tau in enumerate(tau_inds_eval_constraint):
            ax.text(2.3 - (i_spacing * 0.5), 0.01, "{:2.1f}%".format(
                100 * constraints[i_tau, plot_ind - constraint_offset]), ha="right")
            ax.arrow(check_vals[i_tau], 1.05, dx=0, dy=-0.05, length_includes_head=True, color="k", width=0.015,
                     head_width=0.05, head_starts_at_zero=False, ec="k", fc="k", head_length=0.025)
        print("Poisson fraction: {:2.1f}%".format(100 * Poisson_frac[plot_ind]))
    multipage(os.path.join(save_path, "example_constraints.pdf"))
    plt.close("all")
    plt.ion()
