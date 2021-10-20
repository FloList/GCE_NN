"""
Plot calibration and sharpness of the SCD quantiles.
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
sns.set_style("ticks")
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

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    bin_edges = copy.copy(params["gce_hist_bins"])

    out_file = os.path.join(model.get_path("checkpoints"), "calibration_data")

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Get test data
    n_samples_analysis = 1024
    models_test = params["models"]
    model_names = params["model_names"]

    # Define quantile levels tau
    alpha_vec = np.linspace(0.025, 0.475, 19)
    n_alphas = len(alpha_vec)
    tau_vec = np.hstack([0.5 - alpha_vec[::-1], 0.5 + alpha_vec])
    n_taus = len(tau_vec)
    alpha_range_vec = (tau_vec[::-1] - tau_vec)[:n_alphas]
    n_temps = 2

    if not os.path.exists(out_file + ".npz"):
        sess_test = None if TEST_CHECKPOINT is None else model.get_session(checkpoint=TEST_CHECKPOINT)
        input_test_dict = ds_test.next_element.vars()
        test_out = ds_test.get_samples(n_samples_analysis)
        test_data, test_label = test_out["data"], test_out["label"]
        test_gce_hist = test_out["gce_hist"]
        real_fluxes = test_label

        pred_array = np.zeros((n_taus, n_samples_analysis, len(bin_centres), n_temps))

        for i_tau, tau in enumerate(tau_vec):
            hist_pred = model.predict(test_out, sess_test, tau_hist=tau * np.ones((n_samples_analysis, 1)))
            pred_array[i_tau, :, :, :] = hist_pred["gce_hist"][:, :, :]

        pred_array_cum = pred_array.cumsum(2)  # n_taus x n_samples x n_bins x n_temps
        test_gce_hist_cum = test_gce_hist.cumsum(1)  # n_samples x n_bins x n_temps

        # Calculate coverage
        cutoff = 1e-5  # if CDF is < cutoff or > 1 - cutoff: ignore in order not to bias results by irrelevant bins
        tol = 0.0
        coverage = np.zeros((n_alphas, n_temps))
        for i_temp in range(n_temps):
            for i_alpha, alpha in enumerate(alpha_vec):
                within_q_range = np.logical_and(test_gce_hist_cum[:, :, i_temp] >= pred_array_cum[i_alpha, :, :, i_temp] - tol,
                                                test_gce_hist_cum[:, :, i_temp] <= pred_array_cum[n_taus - 1 - i_alpha, :, :, i_temp] + tol)
                outside_inds = np.nonzero(np.logical_or(test_gce_hist_cum[:, :, i_temp] < cutoff,
                                                        test_gce_hist_cum[:, :, i_temp] > 1 - cutoff))
                within_q_range = within_q_range.astype(np.float32)
                within_q_range[outside_inds[0], outside_inds[1]] = np.nan
                coverage[i_alpha, i_temp] = np.nanmean(within_q_range)
        print(coverage)

        np.savez(out_file, test_gce_hist_cum=test_gce_hist_cum, pred_array_cum=pred_array_cum, coverage=coverage)
        print("Done!")

    else:
        data_loaded = np.load(out_file + ".npz")
        coverage = data_loaded["coverage"]
        pred_array_cum = data_loaded["pred_array_cum"]
        test_gce_hist_cum = data_loaded["test_gce_hist_cum"]

    # Plot
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k-", lw=2)
    markers = ["o", "."]
    mfcs = ["white", "k"]
    mecs = ["deepskyblue", "k"]
    mss = [8, 5]
    template_names = ["GCE", "Disk"]
    for i_temp in range(n_temps):
        ax.plot(alpha_range_vec, coverage[:, i_temp], "k", marker=markers[i_temp], lw=0, markersize=mss[i_temp],
                mfc=mfcs[i_temp], markeredgewidth=2, mec=mecs[i_temp], label=template_names[i_temp])
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r"Confidence level $\alpha$")
    ax.set_ylabel(r"Coverage $p_{\mathrm{cov}}(\alpha)$")
    ax.set_aspect("equal")
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # ax.legend()
    plt.tight_layout()
    fig_name = "calibration_plot.pdf"
    fig.savefig(os.path.join(model.get_path("checkpoints"), fig_name), bbox_inches="tight")

    # Compute sharpness
    cutoff = 1e-5
    min_bin = 0
    max_bin = -1
    assert tau_vec[-1] - tau_vec[0] == 0.95
    I95 = pred_array_cum[-1, :, min_bin:max_bin, :] - pred_array_cum[0, :, min_bin:max_bin, :]
    outside_inds = np.nonzero(np.logical_or(test_gce_hist_cum[:, min_bin:max_bin, :] < cutoff,
                                            test_gce_hist_cum[:, min_bin:max_bin, :] > 1 - cutoff))
    I95[outside_inds[0], outside_inds[1], outside_inds[2]] = np.nan
    cols = ["deepskyblue", "k"]
    bins_sha = np.linspace(0, 1, 31)
    label_pos = [[0.45, 9.1], [0.45, 8.6]]
    fig_sha, ax_sha = plt.subplots(1, 1, figsize=(4, 4))
    for i_temp in range(n_temps)[::-1]:
        I95_flat = I95[:, :, i_temp].flatten()
        I95_flat_notnan = I95_flat[~np.isnan(I95_flat)]
        # Make histogram plot
        hist_sha = ax_sha.hist(I95_flat_notnan, bins=bins_sha, density=True, color=cols[i_temp], alpha=0.4)
        sha_val = I95_flat_notnan.mean()
        ax_sha.axvline(sha_val, color=cols[i_temp], ls="--")
        str_sha = r"$\mathcal{S}^{0.95}$ = " + "{:#.2f}".format(sha_val)
        ax_sha.text(*label_pos[i_temp], str_sha, ha="left", va="top", color=cols[i_temp])

    ax_sha.set_xlim([-0.05, 1.05])
    ax_sha.set_ylim([0, 9.5])
    # ax_sha.set_xlabel(r"$\tilde{\mathcal{R}}_j^{\boldsymbol{\varpi}}(\cdot\, ; \, 0.95)$")  # activate tex to render this
    ax_sha.set_xlabel(r"$\tilde{\mathcal{R}}_j^{{\varpi}}(\cdot\, ; \, 0.95)$")
    ax_sha.set_ylabel(r"Normalized frequency")
    plt.tight_layout()
    fig_name_sha = "sharpness.pdf"
    fig_sha.savefig(os.path.join(model.get_path("checkpoints"), fig_name_sha), bbox_inches="tight")

    # Combined plot with iso:
    # fig_comb, axs_comb = plt.subplots(2, 2, figsize=(8, 8), sharex="row", sharey="row")
    # Now: copy paste the respective parts from this script and the Isotropic example calibration script

    # Calibration and sharpness plot for individual bins
    fig_sha_pb, axs_sha_pb = plt.subplots(6, 4, figsize=(7.6, 10))
    axs_sha_pb_flat = axs_sha_pb.flatten()
    fig_cal_pb, axs_cal_pb = plt.subplots(6, 4, figsize=(7.6, 10))
    axs_cal_pb_flat = axs_cal_pb.flatten()
    for i_bin in range(len(axs_sha_pb_flat)):
        if i_bin >= len(bin_centres):
            axs_sha_pb_flat[i_bin].remove()
            axs_cal_pb_flat[i_bin].remove()
            continue
        cutoff = 1e-5
        tol = 0
        I95 = pred_array_cum[-1, :, i_bin:i_bin+1, :] - pred_array_cum[0, :, i_bin:i_bin+1, :]
        outside_inds = np.nonzero(np.logical_or(test_gce_hist_cum[:, i_bin:i_bin+1, :] < cutoff,
                                                test_gce_hist_cum[:, i_bin:i_bin+1, :] > 1 - cutoff))
        I95[outside_inds[0], outside_inds[1], outside_inds[2]] = np.nan
        cols = ["deepskyblue", "k"]
        bins_sha = np.linspace(0, 1, 31)

        coverage_pb = np.zeros((n_alphas, n_temps))

        for i_temp in range(n_temps)[::-1]:
            # Sharpness
            I95_flat = I95[:, :, i_temp].flatten()
            I95_flat_notnan = I95_flat[~np.isnan(I95_flat)]

            # Calibration
            outside_inds_pb = np.nonzero(np.logical_or(test_gce_hist_cum[:, i_bin:i_bin + 1, i_temp] < cutoff,
                                                       test_gce_hist_cum[:, i_bin:i_bin + 1, i_temp] > 1 - cutoff))
            for i_alpha, alpha in enumerate(alpha_vec):
                within_q_range_pb = np.logical_and(test_gce_hist_cum[:, i_bin:i_bin+1, i_temp]
                                                   >= pred_array_cum[i_alpha, :, i_bin:i_bin+1, i_temp] - tol,
                                                   test_gce_hist_cum[:, i_bin:i_bin+1, i_temp]
                                                   <= pred_array_cum[n_taus - 1 - i_alpha, :, i_bin:i_bin+1, i_temp] + tol)
                within_q_range_pb = within_q_range_pb.astype(np.float32)
                within_q_range_pb[outside_inds_pb[0], outside_inds_pb[1]] = np.nan
                coverage_pb[i_alpha, i_temp] = np.nanmean(within_q_range_pb)

            # Make calibration plot
            axs_cal_pb_flat[i_bin].plot(alpha_range_vec, coverage_pb[:, i_temp], c=cols[i_temp], marker=markers[i_temp],
                                        lw=0, markersize=mss[i_temp] - 2 * (i_temp == 0), mfc=mfcs[i_temp],
                                        markeredgewidth=2, mec=mecs[i_temp], label=template_names[i_temp], zorder=2+i_temp)
            axs_cal_pb_flat[i_bin].plot([0, 1], [0, 1], "k-", lw=2)
            n_nonzero = I95_flat.shape[0] - len(outside_inds_pb[0])
            t = axs_cal_pb_flat[i_bin].text(1.0, -0.02 + 0.22 * i_temp, n_nonzero, ha="right", va="bottom", color=mecs[i_temp],
                                            size="small")
            t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='none'))

            # Make sharpness plot
            hist_sha = axs_sha_pb_flat[i_bin].hist(I95_flat_notnan, bins=bins_sha, density=True, color=cols[i_temp], alpha=0.4)
            sha_val = I95_flat_notnan.mean()
            axs_sha_pb_flat[i_bin].axvline(sha_val, color=cols[i_temp], ls="--", alpha=0.5)
            t = axs_sha_pb_flat[i_bin].text(0.1, 19 - 4 * i_temp, n_nonzero, ha="left", va="bottom", color=mecs[i_temp],
                                            size="small")

        # Calibration plot settings
        axs_cal_pb_flat[i_bin].set_xlim([-0.1, 1.1])
        axs_cal_pb_flat[i_bin].set_ylim([-0.1, 1.1])
        # axs_cal_pb_flat[i_bin].set_aspect("equal")
        ticks = np.linspace(0, 1, 6)
        axs_cal_pb_flat[i_bin].set_xticks(ticks)
        axs_cal_pb_flat[i_bin].set_yticks(ticks)
        axs_cal_pb_flat[i_bin].set_xticks([0, 0.5, 1])
        axs_cal_pb_flat[i_bin].set_yticks([0, 0.5, 1])
        axs_cal_pb_flat[i_bin].set_xticklabels(["0", "0.5", "1"])
        axs_cal_pb_flat[i_bin].set_yticklabels(["0", "0.5", "1"])

        # Sharpness plot settings
        axs_sha_pb_flat[i_bin].set_xlim([-0.05, 1.05])
        axs_sha_pb_flat[i_bin].set_ylim([0, 24])
        # axs_sha_pb_flat[i_bin].text(0.95, 18, str(i_bin + 1), ha="right")
        t = axs_sha_pb_flat[i_bin].text(0.97, 19, r"$-$" + "{:2.1f}".format(bin_centres[i_bin])[1:], ha="right")
        t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='none'))
        axs_sha_pb_flat[i_bin].set_xticks([0, 0.5, 1])
        axs_sha_pb_flat[i_bin].set_xticklabels(["0", "0.5", "1"])
        if not axs_sha_pb_flat[i_bin].is_first_col():
            axs_sha_pb_flat[i_bin].set_yticks([])
            axs_cal_pb_flat[i_bin].set_yticks([])
        if not axs_sha_pb_flat[i_bin].is_last_row():
            axs_cal_pb_flat[i_bin].set_xticks([])

    axs_sha_pb_flat[1].set_title("                  Binwise sharpness plot")
    fig_name_sha_pb = "sharpness_binwise.pdf"
    fig_name_cal_pb = "calibration_binwise.pdf"
    plt.figure(fig_sha_pb.number)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.figure(fig_cal_pb.number)
    plt.subplots_adjust(hspace=0, wspace=0)
    fig_sha_pb.savefig(os.path.join(model.get_path("checkpoints"), fig_name_sha_pb), bbox_inches="tight")
    fig_cal_pb.savefig(os.path.join(model.get_path("checkpoints"), fig_name_cal_pb), bbox_inches="tight")

    # Also: compute avg. EMD
    # from scipy.stats import wasserstein_distance
    median_ind_up = n_taus // 2
    median_ind_down = median_ind_up - 1
    pred_array_cum_median = 0.5 * (pred_array_cum[median_ind_up, :, :, :] + pred_array_cum[median_ind_down, :, :, :])
    pred_array_median = np.concatenate([pred_array_cum_median[:, :1, :], np.diff(pred_array_cum_median, axis=1)], axis=1)
    test_gce_hist = np.concatenate([test_gce_hist_cum[:, :1, :], np.diff(test_gce_hist_cum, axis=1)], axis=1)

    pred_array_median_gce = pred_array_median[:, :, 0]
    test_gce_hist_gce = test_gce_hist[:, :, 0]

    EMD = np.zeros((n_samples_analysis, 2))
    for i in range(n_samples_analysis):
        for temp in range(2):
            # EMD[i] = wasserstein_distance(bin_centres, bin_centres, test_gce_hist_gce[i], pred_array_median_gce[i])
            EMD[i, temp] = np.abs(test_gce_hist_cum[i, :, temp] - pred_array_cum_median[i, :, temp]).sum() * np.diff(bin_centres)[0]  # equivalent


    # test_hist_1 = np.asarray([1] + 21 * [0])
    # test_hist_2 = np.asarray([0] + [1] + 20 * [0])
    # np.abs(test_hist_1.cumsum() - test_hist_2.cumsum()).sum() * np.diff(bin_centres)[0]  # -> should give np.diff(bin_centres)[0]
