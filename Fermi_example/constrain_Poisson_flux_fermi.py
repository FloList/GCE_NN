"""
This script computes constraints on the Poissonian flux fraction using the SIMPLE (less powerful) estimator
discussed in arXiv:2107.09070.
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
    TEST_EXP_PATH = "./checkpoints/Fermi_example_add_two_256_BN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
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

    data_folder = os.path.join(model.get_path("checkpoints"), "Best_fit_maps_random_GCE")
    pred_out_folder = os.path.join(data_folder, "Predictions")
    mkdir_p(pred_out_folder)
    save_path = os.path.join(data_folder, "Simple")
    mkdir_p(save_path)
    pred_out_file = "Pred"

    all_taus = np.linspace(0.05, 0.95, 19)
    n_taus = len(all_taus)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    # Check if data exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError("Run 'generate_Fermi_best_fit_maps_random_GCE.py' first!")
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
                         NN_pred_mix_all=NN_pred_mix_all, label=data_loc["label"], label_full=data_loc["label_full"],
                         gce_hist=data_loc["gce_hist"], gce_poiss_ff=data_loc["gce_poiss_ff"])
            print("Predictions for mixed PS/Poisson data SAVED!")
            sys.exit(0)
        else:
            # Load
            NN_pred_mix_all = []
            gce_hist = []
            gce_poiss_ff = []
            for i_m, m in enumerate(map_files):
                NN_pred_data = np.load(os.path.join(pred_out_folder, pred_out_file + "_" + m[:-4]) + ".npz", allow_pickle=True)
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

    # Get predictions for mixed maps
    n_eval = NN_pred_mix_all.shape[1]
    NN_pred_mix_all_cum = NN_pred_mix_all.cumsum(2)

    # Plot the distribution of the Poissonian fraction.
    # If the maps were summed "naively" at random, this should give a triangular distribution,
    # if sorted in reverse order a uniform distribution
    plt.figure()
    plt.hist(gce_poiss_ff, bins=np.linspace(0, 1, 11), density=True)

    # We will mostly work with the median predictions for the mixed / PS data here
    median_ind = (n_taus - 1) // 2
    mix_hist = NN_pred_mix_all[median_ind]
    mix_hist_cum = NN_pred_mix_all_cum[median_ind]

    # Find check_vals such that we get a calibrated estimator
    # only calibrate on a fraction of the data and evaluate on the rest
    n_cal = 4 * n_eval // 5
    n_val = n_eval // 5
    Poisson_frac_constraint = np.zeros((n_taus, n_cal))
    FF_lies_below_value = np.zeros((n_taus, n_cal))
    interp_bins = np.linspace(bin_centres[0], bin_centres[-1], 10001)
    check_vals = np.linspace(-12.5, -10.5, n_taus)  # Initial guess for check_vals
    Poisson_frac = np.squeeze(gce_poiss_ff, -1)  # second dimension can be squeezed

    tol = 0.001
    dt = 0.1
    counter = 1
    max_iter = 1000
    i_tau_c = median_ind
    n_print = 10

    check_vals = np.asarray([-9.78607569, -10.04084573, -10.25873248, -10.45230867,
                             -10.59206651, -10.71615785, -10.80027678, -10.88662787,
                             -10.97650998, -11.03078906, -11.08700197, -11.15246954,
                             -11.23097787, -11.30038243, -11.37820498, -11.47311174,
                             -11.57882489, -11.69455418, -11.83544312])  # this is the result of the below optimisation

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
    #             Poisson_frac_constraint[i_tau, i] = alpha * NN_pred_mix_all_cum[i_tau_c, i, bin_ind_above] \
    #                     + (1 - alpha) * NN_pred_mix_all_cum[i_tau_c, i, bin_ind_below]
    #             if Poisson_frac[i] <= Poisson_frac_constraint[i_tau, i]:
    #                 FF_lies_below_value[i_tau, i] = 1
    #
    #     update = all_taus[::-1] - FF_lies_below_value.mean(1)
    #     check_vals = check_vals + dt * update
    #     check_vals.sort()
    #     check_vals = check_vals[::-1]
    #     if counter % n_print == 0:
    #         print("  Mean cal. error:", np.abs(update).mean())
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
        alpha = (check_vals[i_tau] - bin_centres[bin_ind_below]) / (bin_centres[bin_ind_above] - bin_centres[bin_ind_below])
        for i in range(n_cal, n_eval):
            constraint = alpha * NN_pred_mix_all_cum[i_tau_c, i, bin_ind_above] \
                                                     + (1 - alpha) * NN_pred_mix_all_cum[i_tau_c, i, bin_ind_below]
            constraints[i_tau, i - n_cal] = constraint
        print("tau:{:1.2f}, coverage:{:1.4f}".format(all_taus[::-1][i_tau], (constraints[i_tau] > Poisson_frac[n_cal:n_eval]).mean()))

    # Make a calibration plot for the unseen data
    fig_cal, ax_cal = plt.subplots(1, 1, figsize=(3.5, 3.5))
    ax_cal.plot(all_taus[::-1], all_taus[::-1], "k-", lw=1)
    ax_cal.plot(all_taus[::-1], (constraints > Poisson_frac[n_cal:n_eval]).mean(1),
                "ko", lw=0, markersize=8, mfc="white", markeredgewidth=2)
    fig_cal.savefig(os.path.join(save_path, "calibration_plot_uncalibrated_data.pdf"), bbox_inches="tight")

    # Plot the check_vals
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(all_taus[::-1], check_vals, "ko", lw=0, markersize=8, mfc="white", markeredgewidth=2)
    plt.xlabel("Confidence level")
    plt.ylabel(r"$\phi$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "phi_vs_alpha.pdf"), bbox_inches="tight")

    # Plot some examples (from the dataset it was not calibrated on)
    tau_inds_eval_constraint = [0, 5, 9]  # NOTE: THE ORDER IS REVERSED, SO THE INDICES CORRESPOND TO 1 - tau!!! for example [0, 5, 9] -> 95%, 70%, 50%
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
                                    y2=NN_pred_mix_all_cum[i_tau + 1, plot_ind, i], color=colors[i_tau], lw=0, alpha=0.5)
        ax.plot(bin_centres, gce_hist[plot_ind, :, 0].cumsum(), color="darkslateblue", ls="-", lw=2)  # True label (PS only, 0: GCE)
        ax.axhline(Poisson_frac[plot_ind], color="k", lw=1, ls="--")
        ax.text(-11, 0.05, "True Poisson FF:", ha="left")
        ax.text(-7, 0.05, "{:2.1f}%".format(100 * Poisson_frac[plot_ind]), ha="right")
        ax.text(-11, 0.01, "Constraints:", ha="left")
        for i_spacing, i_tau in enumerate(tau_inds_eval_constraint):
            ax.text(-7 - (i_spacing * 0.8), 0.01, "{:2.1f}%".format(100 * constraints[i_tau, plot_ind - constraint_offset]),
                    ha="right")
            ax.arrow(check_vals[i_tau], 1.05, dx=0, dy=-0.05, length_includes_head=True, color="k", width=0.015,
                     head_width=0.05, head_starts_at_zero=False, ec="k", fc="k", head_length=0.025)
        print("Poisson fraction: {:2.1f}%".format(100 * Poisson_frac[plot_ind]))
    multipage(os.path.join(save_path, "example_constraints.pdf"))
    plt.close("all")
    plt.ion()

    # NOW: Apply to Fermi map!
    fermi_pred_data = np.load(os.path.join(model.get_path("checkpoints"), "fermi_prediction.npz"), allow_pickle=True)
    print(fermi_pred_data.files)
    fermi_pred = fermi_pred_data["fermi_pred"][()]
    fermi_pred_hist_gce_cum = fermi_pred["gce_hist"][:, :, 0].cumsum(1)

    fermi_constraints = np.zeros((n_taus, n_taus))  # first index: quantile levels for Fermi prediction
                                                    # second index: confidence w.r.t. FF constraint
    for i_tau in range(len(all_taus)):
        bin_ind_above = np.argmax(bin_centres > check_vals[i_tau])
        bin_ind_below = bin_ind_above - 1
        alpha = (check_vals[i_tau] - bin_centres[bin_ind_below]) / (bin_centres[bin_ind_above] - bin_centres[bin_ind_below])
        for i_tau_fermi in range(len(all_taus)):
            fermi_constraint = alpha * fermi_pred_hist_gce_cum[i_tau_fermi][bin_ind_above] + (1 - alpha) * fermi_pred_hist_gce_cum[i_tau_fermi][bin_ind_below]
            fermi_constraints[i_tau_fermi, i_tau] = fermi_constraint

    fig_fermi, ax_fermi = plt.subplots(1, 1, figsize=(6, 6))
    im = ax_fermi.imshow(100 * np.flipud(fermi_constraints.T), cmap=cc.cm.CET_D3_r, aspect='equal', interpolation='none', vmin=0, vmax=100, origin='lower')
    cbar = fig_fermi.colorbar(im, fraction=0.0458, pad=0.04)
    cbar.set_label(r"Max. Poisson flux fraction [$\%$]")
    ax_fermi.set_xlabel(r"Quantile level $\tau$")
    ax_fermi.set_ylabel("Confidence")
    ticks = [0, 3, 6, 9, 12, 15, 18]
    tick_labels = ["{:#1.2f}".format(t) for t in np.round(all_taus[ticks], 2)]
    ax_fermi.set_xticks(ticks)
    ax_fermi.set_xticklabels(tick_labels)
    ax_fermi.set_yticks(ticks)
    ax_fermi.set_yticklabels(tick_labels)
    rect = mpl.patches.Rectangle([median_ind - 0.5, ax_fermi.get_ylim()[0]], 1.04, np.diff(ax_fermi.get_ylim()), fill=False,
                                 ec="k", lw=2)
    ax_fermi.add_patch(rect)
    plt.tight_layout()
    fig_fermi.savefig(os.path.join(save_path, "Fermi_constraints.pdf"), bbox_inches="tight")
