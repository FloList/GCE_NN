"""
Plot calibration and sharpness of the SCD quantiles.
"""
import os
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
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
    TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
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

    if not os.path.exists(out_file + ".npz"):
        sess_test = None if TEST_CHECKPOINT is None else model.get_session(checkpoint=TEST_CHECKPOINT)
        input_test_dict = ds_test.next_element.vars()
        test_out = ds_test.get_samples(n_samples_analysis)
        test_data, test_label = test_out["data"], test_out["label"]
        test_gce_hist = test_out["gce_hist"]
        real_fluxes = test_label

        pred_array = np.zeros((n_taus, n_samples_analysis, len(bin_centres)))

        for i_tau, tau in enumerate(tau_vec):
            hist_pred = model.predict(test_out, sess_test, tau_hist=tau * np.ones((n_samples_analysis, 1)))
            pred_array[i_tau, :, :] = hist_pred["gce_hist"][:, :, 0]

        pred_array_cum = pred_array.cumsum(2)
        test_gce_hist_cum = test_gce_hist.cumsum(1)

        # Calculate coverage
        cutoff = 1e-5  # if CDF is < cutoff or > 1 - cutoff: ignore in order not to bias results by irrelevant bins
        tol = 0.0
        coverage = np.zeros(n_alphas)
        for i_alpha, alpha in enumerate(alpha_vec):
            within_q_range = np.logical_and(test_gce_hist_cum[:, :, 0] > pred_array_cum[i_alpha, :, :] - tol,
                                            test_gce_hist_cum[:, :, 0] < pred_array_cum[n_taus - 1 - i_alpha, :,
                                                                         :] + tol)
            outside_inds = np.nonzero(np.logical_or(test_gce_hist_cum[:, :, 0] < cutoff,
                                                    test_gce_hist_cum[:, :, 0] > 1 - cutoff))
            within_q_range = within_q_range.astype(np.float32)
            within_q_range[outside_inds[0], outside_inds[1]] = np.nan
            coverage[i_alpha] = np.nanmean(within_q_range)
        print(coverage)

        np.savez(out_file, test_gce_hist_cum=test_gce_hist_cum, pred_array_cum=pred_array_cum, coverage=coverage)
        print("Done!")

    else:
        data_loaded = np.load(out_file + ".npz")
        coverage = data_loaded["coverage"]
        pred_array_cum = data_loaded["pred_array_cum"]
        test_gce_hist_cum = data_loaded["test_gce_hist_cum"]

    # Calibration plot
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k-", lw=2)
    ax.plot(alpha_range_vec, coverage, "ko", lw=0, markersize=8, mfc="white", markeredgewidth=2)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(r"Confidence level $\alpha$")
    ax.set_ylabel(r"Coverage $p_{\mathrm{cov}}(\alpha)$")
    ax.set_aspect("equal")
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.tight_layout()
    fig_name = "calibration_plot.pdf"
    fig.savefig(os.path.join(model.get_path("checkpoints"), fig_name), bbox_inches="tight")

    # Compute sharpness
    cutoff = 1e-5
    assert tau_vec[-1] - tau_vec[0] == 0.95
    I95 = pred_array_cum[-1, :, :] - pred_array_cum[0, :, :]
    outside_inds = np.nonzero(np.logical_or(test_gce_hist_cum[:, :, 0] < cutoff, test_gce_hist_cum[:, :, 0] > 1 - cutoff))
    I95[outside_inds[0], outside_inds[1]] = np.nan
    I95_flat = I95.flatten()
    I95_flat_notnan = I95_flat[~np.isnan(I95_flat)]
    # Make histogram plot
    bins_sha = np.linspace(0, 1, 31)
    fig_sha, ax_sha = plt.subplots(1, 1, figsize=(4, 4))
    hist_sha = ax_sha.hist(I95_flat_notnan, bins=bins_sha, density=True, color="k", alpha=0.4)
    sha_val = I95_flat_notnan.mean()
    ax_sha.axvline(sha_val, color="k", ls="--")
    str_sha = r"$\mathcal{S}^{0.95}$ = " + "{:#.3f}".format(sha_val)
    ax_sha.text(sha_val + 0.025, ax_sha.get_ylim()[1] * 0.95, str_sha, ha="left", va="top")
    ax_sha.set_xlim([-0.05, 1.05])
    #ax_sha.set_xlabel(r"$\tilde{\mathcal{R}}_j^{\boldsymbol{\varpi}}(\cdot\, ; \, 0.95)$")  # activate tex to render this
    ax_sha.set_xlabel(r"$\tilde{\mathcal{R}}_j^{{\varpi}}(\cdot\, ; \, 0.95)$")
    ax_sha.set_ylabel(r"Normalized frequency")
    plt.tight_layout()
    fig_name_sha = "sharpness.pdf"
    fig_sha.savefig(os.path.join(model.get_path("checkpoints"), fig_name_sha), bbox_inches="tight")

    # Also: compute avg. EMD
    # from scipy.stats import wasserstein_distance
    median_ind_up = n_taus // 2
    median_ind_down = median_ind_up - 1
    pred_array_cum_median = 0.5 * (pred_array_cum[median_ind_up, :, :] + pred_array_cum[median_ind_down, :, :])

    EMD = np.zeros(n_samples_analysis)
    for i in range(n_samples_analysis):
            EMD[i] = np.abs(test_gce_hist_cum[i, :, 0] - pred_array_cum_median[i, :]).sum() * np.diff(bin_centres)[0]  # equivalent to scipy.stats.wasserstein_distance
    print(EMD)
