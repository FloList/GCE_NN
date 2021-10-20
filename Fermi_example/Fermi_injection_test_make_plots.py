"""
This script produces plots for the Fermi GCE flux injection experiment (both DM and PS).
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from matplotlib.ticker import LogLocator
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


def F2S(x):
    return 10.0 ** x * mean_exp


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
    # TEST_EXP_PATH = "./checkpoints/Fermi_example_add_two_BN_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    np.random.seed(0)
    tf.random.set_seed(0)

    data_out_folder_PS = "Fermi_injection_data_PS"
    data_out_file_PS = "Fermi_injection_data_PS"
    data_out_file_all_PS = "Fermi_injection_data_PS_all"
    data_out_file_DM = "Fermi_injection_data"

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    bin_edges = copy.copy(params["gce_hist_bins"])
    nside = params["nside"]

    fermi_folder = get_fermi_folder_basename(GADI, w573=True)
    fermi_folder += "/fermi_data_" + str(nside)

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Get Fermi map prediction
    try:
        fermi_pred_data = np.load(os.path.join(model.get_path("checkpoints"), "fermi_prediction.npz"),
                                  allow_pickle=True)
        print(fermi_pred_data.files)
        fermi_pred = fermi_pred_data["fermi_pred"][()]
        bin_centres = fermi_pred_data["bin_centres"][()]
        tau_vec = fermi_pred_data["tau_vec"][()]
    except FileNotFoundError:
        raise FileNotFoundError("Run the script 'save_fermi_prediction' first!")

    # Get FFs and histogram
    total_fermi_counts = (generator_test.settings_dict["rescale"] * fermi_counts).sum()  # total counts in ROI
    exp_compressed_nest = generator_test.settings_dict["exp"][generator_test.settings_dict["unmasked_pix"]]
    total_fermi_flux = (generator_test.settings_dict["rescale"] * fermi_counts / exp_compressed_nest).sum()
    FFs = fermi_pred["logits_mean"].mean(0)  # avg. over taus (identical)
    fluxes = FFs * total_fermi_flux

    # We have:
    #     f^new_t   = f^old_t     + Delta f_t
    #     f^new_tot = f^old_tot   + Delta f_t
    #     Delta f_t = xi * f^new_tot
    #     Delta f_t = xi / (1 - xi) * f^old_tot

    mean_exp = generator_test.settings_dict["exp"].mean()
    n_maps = 64  # maps for each FF and PS brightness
    xi_vec = np.linspace(0, 0.08, 9)[1:]  # FFs
    f_vec = np.logspace(-1, 1, 5) / mean_exp  # flux per PS array
    n_xi = len(xi_vec)
    n_f = len(f_vec)

    # Define quantile levels tau
    n_taus = len(tau_vec)

    # Load PS injection data
    inj_data_PS = np.load(os.path.join(model.get_path("checkpoints"), data_out_folder_PS, data_out_file_all_PS + ".npz"),
                          allow_pickle=True)
    all_FFs_inj_PS = inj_data_PS["all_FFs_inj"][()]
    all_FF_stds_inj_PS = inj_data_PS["all_FF_stds_inj"][()]
    all_hists_inj_PS = inj_data_PS["all_hists_inj"][()]

    # Load DM injection data
    inj_data_DM = np.load(os.path.join(model.get_path("checkpoints"), data_out_file_DM + ".npz"), allow_pickle=True)
    all_FFs_inj_DM = inj_data_DM["all_FFs_inj"][()]
    all_FF_stds_inj_DM = inj_data_DM["all_FF_stds_inj"][()]
    all_hists_inj_DM = inj_data_DM["all_hists_inj"][()]

    # 1) Make a plot of the injected vs recovered GCE flux fractions
    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    fig_ff, ax_ff = plt.subplots(1, 1, figsize=(4, 4))
    GCE_ind = 4

    # Line for truth:
    # at xi = 0:  (xi, FF) = (0, FP), where FP = Fermi prediction (without injection)
    # at max_xi:  (xi, FF) = (max_xi, (GCE flux orig + Delta flux) / (Total flux orig + Delta flux))
    max_xi = 0.08
    GCE_flux_orig = FFs[GCE_ind] * total_fermi_flux
    Delta_f_xi_max = total_fermi_flux * max_xi / (1 - max_xi)
    exp_FF_xi_max = (GCE_flux_orig + Delta_f_xi_max) / (total_fermi_flux + Delta_f_xi_max)
    ax_ff.plot([0, 100 * max_xi], [100 * FFs[GCE_ind], 100 * exp_FF_xi_max], "k--", lw=1, alpha=0.4)

    # Prediction for Fermi map
    ax_ff.plot(0, 100 * FFs[GCE_ind], ls="none", marker="o", ms=5, color="k", mfc="white", mew=2)

    # Slightly shift in x-direction for each case
    shift_x = 0.15
    shift_0 = - (len(f_vec) + 1) // 2 * shift_x
    colors_inj = cc.cm.fire(np.linspace(0, 1, len(f_vec) + 3))[1:-1]
    shift_x_line_length = (len(f_vec) + 2) / 2 * shift_x

    # Draw horizontal lines at correct values
    Delta_f_xis = total_fermi_flux * xi_vec / (1 - xi_vec)
    exp_FF_xis = (GCE_flux_orig + Delta_f_xis) / (total_fermi_flux + Delta_f_xis)
    for i_xi, xi in enumerate(xi_vec):
        ax_ff.plot([100 * xi - shift_x_line_length, 100 * xi + shift_x_line_length], 2 * [100 * exp_FF_xis[i_xi]], ls="-",
                   color="k", lw=1, alpha=0.8)

    # First: DM
    FFs_inj_GCE_DM = all_FFs_inj_DM[:, :, GCE_ind]
    FF_median_DM = np.median(FFs_inj_GCE_DM, axis=1)
    yerr_low_DM = np.quantile(FFs_inj_GCE_DM, 0.5, axis=1) - np.quantile(FFs_inj_GCE_DM, 0.16, axis=1)
    yerr_high_DM = np.quantile(FFs_inj_GCE_DM, 0.84, axis=1) - np.quantile(FFs_inj_GCE_DM, 0.5, axis=1)
    yerr_DM = np.vstack([yerr_low_DM, yerr_high_DM])
    ax_ff.errorbar(100 * np.asarray(xi_vec) + shift_0 + shift_x / 2, y=100 * FF_median_DM, ls="none", yerr=100 * yerr_DM,
                   capsize=2, ecolor="k", marker=".", ms=0, markeredgewidth=1, elinewidth=2, mec="deepskyblue",
                   mfc="white", barsabove=True)

    for i_f in range(len(f_vec)):

        FFs_inj_GCE = all_FFs_inj_PS[:, :, i_f, GCE_ind]

        # Predictions for injection maps
        FF_median = np.median(FFs_inj_GCE, axis=1)
        yerr_low = np.quantile(FFs_inj_GCE, 0.5, axis=1) - np.quantile(FFs_inj_GCE, 0.16, axis=1)
        yerr_high = np.quantile(FFs_inj_GCE, 0.84, axis=1) - np.quantile(FFs_inj_GCE, 0.5, axis=1)
        yerr = np.vstack([yerr_low, yerr_high])
        ax_ff.errorbar(100 * np.asarray(xi_vec) + (i_f + 1.5) * shift_x + shift_0, y=100 * FF_median, ls="none", yerr=100 * yerr,
                       capsize=2, ecolor=colors_inj[i_f], marker=".", ms=0, markeredgewidth=1, elinewidth=2, barsabove=True)

    # Plot injected FF that corresponds to upper prior limit
    upper_prior_limit_DM_flux = 2.8e-07
    upper_prior_limit_DM_counts = upper_prior_limit_DM_flux * generator_test.settings_dict["exp"].mean()
    Delta_f_lim = upper_prior_limit_DM_flux - GCE_flux_orig
    Delta_xi_lim = Delta_f_lim / (Delta_f_lim + total_fermi_flux)
    # Check correctness
    print("Delta_xi_lim:", Delta_xi_lim)
    print("This gives:", (Delta_xi_lim / (1 - Delta_xi_lim) * total_fermi_flux + GCE_flux_orig) * generator_test.settings_dict["exp"].mean(),
          "GCE counts for the injection map at Delta xi lim.")
    rect = mpl.patches.Rectangle([100 * Delta_xi_lim, -0.5 + 100 * FFs[GCE_ind]], 100 * exp_FF_xi_max + 0.5 - Delta_xi_lim,
                                 100 * exp_FF_xi_max - 100 * FFs[GCE_ind] + 1, fill=True, color="k", ec="none", alpha=0.2)
    ax_ff.add_patch(rect)
    # ax.axvline(100 * Delta_xi_lim, color="k", alpha=0.3, linestyle=":")
    ax_ff.set_xlim([-0.5, 100 * max_xi + 0.5])
    ax_ff.set_ylim([-0.5 + 100 * FFs[GCE_ind], 100 * exp_FF_xi_max + 0.5])
    ax_ff.set_aspect("equal")
    ax_ff.set_xlabel(r"Injected GCE flux fraction [%]")
    ax_ff.set_ylabel(r"Recovered GCE flux fraction [%]")
    plt.tight_layout()
    fig_name = "Fermi_DM_and_PS_injection_FF_plot.pdf"
    fig_ff.savefig(os.path.join(model.get_path("checkpoints"), fig_name), bbox_inches="tight")

    # 2) Plot for the predicted histograms
    # Make a big plot: in each row: first, densities, at the end: one cumul. plot (only median, all FFs)
    xi_inds_to_plot = [1, 3, 5]
    n_xis_to_plot = len(xi_inds_to_plot)
    fig_hist, axs_hist = plt.subplots(len(f_vec) + 1, n_xis_to_plot + 1, figsize=(8, 10))
    plt.subplots_adjust(wspace=0, hspace=0)
    ylims = [-0.075, 1.075]
    xlims = [-13, -9]

    # First: Plot CDFs (median only) in the last column
    # Start with DM
    width = np.diff(bin_centres)[0]
    median_ind = all_hists_inj_DM.shape[1] // 2
    colors_xi = cc.cm.bgy(np.linspace(0, 1, n_xi + 2))[:-1]

    for i_xi in range(len(xi_vec) - 1, -1, -1):
        axs_hist[0, -1].step(bin_centres - width / 2, np.median(all_hists_inj_DM[i_xi, median_ind, :, :, 0].cumsum(-1), 0),
                               where="post", color=colors_xi[i_xi + 1], lw=1.5)

    # Plot CDF for original Fermi map
    axs_hist[0, -1].step(bin_centres - width / 2, fermi_pred["gce_hist"][median_ind, :, 0].cumsum(), where="post",
                         color="k", lw=1.5)

    # Do the same for PS injection
    for i_f in range(len(f_vec)):
        for i_xi in range(len(xi_vec) - 1, -1, -1):
            axs_hist[1 + i_f, -1].step(bin_centres - width / 2,
                                       np.median(all_hists_inj_PS[i_xi, i_f, median_ind, :, :, 0].cumsum(-1), 0),
                                       where="post", color=colors_xi[i_xi + 1], lw=1.5)

        # Plot CDF for original Fermi map
        axs_hist[1 + i_f, -1].step(bin_centres - width / 2, fermi_pred["gce_hist"][median_ind, :, 0].cumsum(),
                                   where="post", color="k", lw=1.5)

    # Plot settings
    one_ph_flux = np.log10(1 / mean_exp)
    for i_ax in range(n_f + 1):
        axs_hist[i_ax, -1].set_xlim(xlims)
        axs_hist[i_ax, -1].axvline(one_ph_flux, color="orange", ls="--", zorder=4, lw=1.5)
        rect = mpl.patches.Rectangle((np.log10(4e-10), -0.075), np.log10(5e-10) - np.log10(4e-10), 1.075 + 0.075,
                                     linewidth=0, edgecolor=None, facecolor="#cccccc", zorder=-1)
        axs_hist[i_ax, -1].add_patch(rect)

        if i_ax < n_f:
            axs_hist[i_ax, -1].set_xticks([])
        else:
            axs_hist[i_ax, -1].set_xticks([-12, -11, -10])
        axs_hist[i_ax, -1].set_yticks([])

        if i_ax == 0:
            # Build twin axes and set limits
            twin_axes = axs_hist[0, -1].twiny()
            twin_axes.plot(F2S(bin_centres), fermi_pred["gce_hist"][median_ind, :, 0].cumsum(), color="none", lw=0)
            twin_axes.set_xlim(F2S(np.asarray(xlims)))
            twin_axes.set_xlabel(r"$\bar{S}$")
            twin_axes.set_xscale("log")
            twin_axes.set_xticks(np.logspace(-2, 1, 4))
            twin_axes.set_ylim(ylims)
            twin_axes.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10))
            # axs_hist[0, -1].set_xlabel(r"$\log_{10} \ F$")
            # axs_hist[0, -1].set_ylabel("CDF")
            # plt.tight_layout()

    # Now: Plot the PDFs for all the quantiles, for selected values of xi
    pred_hists_all = np.concatenate([np.expand_dims(all_hists_inj_DM, 1), all_hists_inj_PS], axis=1)[xi_inds_to_plot, :, :, :, :, 0]
    colors_pdf = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]
    color_no_inj = "#00daed"

    for i_f in range(pred_hists_all.shape[1]):
        for i_xi in range(n_xis_to_plot):

            # Get hist
            pred_hist = pred_hists_all[i_xi, i_f]

            # Iterate over the taus
            for i_tau in range(n_taus):

                pred_hist_cum = pred_hist[:, :, :].cumsum(axis=2)
                pred_hist_cum_median = np.median(pred_hist_cum, 1)  # Median over the maps
                pred_hist_median = pred_hist_cum_median[:, 1:] - pred_hist_cum_median[:, :-1]
                pred_hist_median = np.concatenate([pred_hist_cum_median[:, :1], pred_hist_median], axis=1)
                pred_hist_median[pred_hist_median < 0] = 0.0  # avoid neg. values due to numerical errors

                # Plot differential histogram
                axs_hist[i_f, i_xi].fill_between(bin_centres - width / 2.0, pred_hist_median[i_tau, :],
                                                 color=colors_pdf[i_tau], zorder=1, alpha=0.075, step="post")

                # For median: plot a solid line
                if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                    axs_hist[i_f, i_xi].step(bin_centres - width / 2.0, pred_hist_median[i_tau, :], color="k", lw=1.5,
                                             zorder=3, alpha=1.0, where="post")

            # 3FGL
            axs_hist[i_f, i_xi].axvline(one_ph_flux, color="orange", ls="--", lw=1.5)

            # Plot Fermi histogram without injection faint on top
            axs_hist[i_f, i_xi].step(bin_centres - width / 2.0, fermi_pred["gce_hist"][median_ind, :, 0],
                                     color=color_no_inj, lw=1.5, ls="--", zorder=4, alpha=1.0, where="post")

            # Plot line at injected exp. count number
            if i_f > 0:
                # axs_hist[i_f, i_xi].axvline(np.log10(f_vec[i_f - 1]), color=colors_inj[i_f], ls="--", lw=1.5)
                axs_hist[i_f, i_xi].arrow(np.log10(f_vec[i_f - 1]), 1.05, dx=0, dy=-0.15, length_includes_head=True,
                                          color=colors_inj[i_f], width=0.08, head_width=0.2, head_starts_at_zero=False,
                                          ec="k", head_length=0.07, lw=1.0, zorder=6)

    # Adjust plot
    for i_f in range(pred_hists_all.shape[1]):
        for i_xi in range(n_xis_to_plot):
            # Set axes limits
            axs_hist[i_f, i_xi].set_ylim(ylims)
            y_ticks = np.linspace(0, 1, 6)
            axs_hist[i_f, i_xi].set_yticks(y_ticks)
            if i_f == n_f:
                x_ticks = [-12, -11, -10]
                axs_hist[i_f, i_xi].set_xticks(x_ticks)
                x_ticklabels = [r"$" + str(int(t)) + "$" for t in x_ticks]
                axs_hist[i_f, i_xi].set_xticklabels(x_ticklabels)
                axs_hist[i_f, i_xi].set_xlabel(r"$\log_{10} \ F$")
            else:
                axs_hist[i_f, i_xi].set_xticks([])
            axs_hist[i_f, i_xi].set_xlim(xlims)
            axs_hist[i_f, i_xi].set_title("")

            if not i_xi == 0:
                axs_hist[i_f, i_xi].set_yticks([])

    # Upper x axis with expected counts
    twin_axes = [None] * n_xis_to_plot
    for i_xi in range(n_xis_to_plot):
        twin_axes[i_xi] = axs_hist[0, i_xi].twiny()
        twin_axes[i_xi].set_xlabel(r"$\bar{S}$")
        twin_axes[i_xi].set_xscale("log")
        twin_axes[i_xi].set_xlim(F2S(np.asarray(xlims)))
        tick_locs_logS = np.logspace(-2, 1, 4)
        twin_axes[i_xi].set_xticks(tick_locs_logS)
        twin_x_ticklabels = [r"$10^{" + str(int(t)) + "}$" for t in np.linspace(-2, 1, 4)]
        twin_x_ticklabels_final = twin_x_ticklabels
        twin_axes[i_xi].set_xticklabels(twin_x_ticklabels_final)
        locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
        twin_axes[i_xi].xaxis.set_minor_locator(locmin)

    # Draw 3FGL detection threshold -0.075, 1.075
    for i_f in range(pred_hists_all.shape[1]):
        for i_xi in range(n_xis_to_plot):
            rect = mpl.patches.Rectangle((np.log10(4e-10), -0.075), np.log10(5e-10) - np.log10(4e-10), 1.075 + 0.075,
                                         linewidth=0, edgecolor=None, facecolor="#cccccc", zorder=-1)
            axs_hist[i_f, i_xi].add_patch(rect)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    fig_name = "Fermi_DM_and_PS_injection_hist_plot.pdf"
    fig_hist.savefig(os.path.join(model.get_path("checkpoints"), fig_name), bbox_inches="tight")


# Calculate constraint expected from extrapolating FF in Fermi map
original_GCE_flux = total_fermi_flux * 0.079
constraint_eta_P = 0.656  # 0.656 at 95% confidence
original_constraint_DM_flux = original_GCE_flux * constraint_eta_P
xi_vec_wrt_pre_injection = xi_vec / (1 - xi_vec)

# for DM injection
injected_DM_flux = xi_vec_wrt_pre_injection * total_fermi_flux
original_constraint_DM_flux_plus_injected_DM_flux = original_constraint_DM_flux + injected_DM_flux
original_GCE_flux_plus_injected_DM_flux = original_GCE_flux + injected_DM_flux
extrapolated_constraint_for_DM = original_constraint_DM_flux_plus_injected_DM_flux \
                                 / original_GCE_flux_plus_injected_DM_flux

# for PS injection:
injected_PS_flux = xi_vec_wrt_pre_injection * total_fermi_flux
original_GCE_flux_plus_injected_PS_flux = original_GCE_flux + injected_PS_flux
extrapolated_constraint_for_PS = original_constraint_DM_flux / original_GCE_flux_plus_injected_PS_flux
