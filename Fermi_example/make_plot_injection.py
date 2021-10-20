"""
Plotting function for the injection experiment.
"""
import numpy as np
import seaborn as sns
import colorcet as cc
import matplotlib as mpl
from matplotlib.ticker import LogLocator
from matplotlib import pyplot as plt


def sep_comma(s):
    return f'{s:,}'


def make_plot_injection(pred_hist_all, tau_vec, tau_inds_ebars, bin_centres, save_name="injection_plot.pdf", mean_exp=0,
                        width=None, scatter_for_first_map=False):
    # Shape of pred_hist_all:
    # n_xis  x  n_taus  x  n_maps  x  n_bins  x  2 (GCE, disk)
    n_xis, n_taus, n_maps, n_bins, _ = pred_hist_all.shape

    # Get width of bins
    if width is None:
        width = min(np.diff(bin_centres))

    # Get number of quantile levels
    n_taus = len(tau_vec)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    fig, axs = plt.subplots(2, n_xis, figsize=(10.6, 6.4), constrained_layout=True)

    for i_xi in range(n_xis):

        # Get hist
        pred_hist = pred_hist_all[i_xi]

        twin_axes = [None] * 2

        def F2S(x):
            return 10.0 ** x * mean_exp

        # For GCE histogram only
        for i_ch in range(1):

            # Iterate over the taus
            for i_tau in range(n_taus):

                pred_hist_cum = pred_hist[:, :, :, i_ch].cumsum(axis=2)
                pred_hist_cum_median = np.median(pred_hist_cum, 1)  # Median over the maps
                pred_hist_median = pred_hist_cum_median[:, 1:] - pred_hist_cum_median[:, :-1]
                pred_hist_median = np.concatenate([pred_hist_cum_median[:, :1], pred_hist_median], axis=1)
                pred_hist_median[pred_hist_median < 0] = 0.0  # avoid neg. values due to numerical errors

                # For selected indices: plot error bars over the samples
                if i_tau in tau_inds_ebars and (i_xi > 0 or scatter_for_first_map):
                    yerr_low = np.quantile(pred_hist_cum[i_tau], 0.5, axis=0) - np.quantile(pred_hist_cum[i_tau], 0.16, axis=0)
                    yerr_high = np.quantile(pred_hist_cum[i_tau], 0.84, axis=0) - np.quantile(pred_hist_cum[i_tau], 0.5, axis=0)
                    yerr = np.vstack([yerr_low, yerr_high])
                    axs[0, i_xi].errorbar(bin_centres, y=np.quantile(pred_hist_cum[i_tau], 0.5, axis=0), ls="none",
                                          yerr=yerr, capsize=3, ecolor="#aaa9ad", marker="o", color=colors[i_tau], mec="#aaa9ad",
                                          ms=4, markeredgewidth=1, elinewidth=2)

                # Plot differential histogram
                axs[1, i_xi].fill_between(bin_centres - width / 2.0, pred_hist_median[i_tau, :], color=colors[i_tau],
                                          zorder=1, alpha=0.075, step="post")

                # For median: plot a solid line
                if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                    axs[1, i_xi].step(bin_centres - width / 2.0, pred_hist_median[i_tau, :], color="k", lw=2,
                                      zorder=3, alpha=1.0, where="post")

                # Plot cumulative histogram
                if i_tau < n_taus - 1:
                    # Draw the next section of the cumulative median histogram in the right colour
                    for i in range(len(bin_centres)):
                        # Draw the next section of the cumulative histogram in the right colour
                        axs[0, i_xi].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                                  y1=pred_hist_cum_median[i_tau, :][i],
                                                  y2=pred_hist_cum_median[i_tau + 1, :][i], color=colors[i_tau], lw=0)
                        # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                        if i_tau == 0 and pred_hist_cum_median[0, :][i] > 0.99:
                            axs[0, i_xi].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                              color=colors[0], lw=2, zorder=3)
                        elif i_tau == n_taus - 2 and pred_hist_cum_median[-1, :][i] < 0.01:
                            axs[0, i_xi].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                              color=colors[-1], lw=2, zorder=3)

            one_ph_flux = np.log10(1 / mean_exp)
            for i_ax in range(2):
                axs[i_ax, i_xi].axvline(one_ph_flux, color="orange", ls="--")

    # Adjust plot
    for i_xi in range(n_xis):
        for i_ax in range(2):
            # Set axes limits
            axs[i_ax, i_xi].set_ylim([-0.075, 1.075])
            axs[i_ax, i_xi].set_ylim([-0.075, 1.075])
            y_ticks = np.linspace(0, 1, 6)
            axs[i_ax, i_xi].set_yticks(y_ticks)
            if i_ax == 1:
                x_ticks = np.linspace(-13, -8, 6)
                axs[i_ax, i_xi].set_xticks(x_ticks)
                x_ticklabels = [r"$" + str(int(t)) + "$" for t in x_ticks]
                x_ticklabels[0] = ""
                x_ticklabels[-1] = ""
                axs[i_ax, i_xi].set_xticklabels(x_ticklabels)
                axs[i_ax, i_xi].set_xlabel(r"$\log_{10} \ F$")
            else:
                axs[i_ax, i_xi].set_xticks([])
            axs[i_ax, i_xi].set_xlim([-13, -8])
            axs[i_ax, i_xi].set_title("")
            if not i_xi == 0:
                axs[i_ax, i_xi].set_yticks([])

    # Upper x axis with expected counts
    for i_xi in range(n_xis):
        twin_axes[i_ch] = axs[0, i_xi].twiny()
        twin_axes[i_ch].set_xlabel(r"$\bar{S}$")
        twin_axes[i_ch].set_xscale("log")
        twin_axes[i_ch].set_xlim(F2S(np.asarray([-13, -9])))
        tick_locs_logS = np.logspace(-2, 3, 6)
        twin_axes[i_ch].set_xticks(tick_locs_logS)
        twin_x_ticklabels = [r"$10^{" + str(int(t)) + "}$" for t in np.linspace(-2, 3, 6)]
        twin_x_ticklabels_final = twin_x_ticklabels if i_xi == n_xis - 1 else twin_x_ticklabels[:-1] + [""]
        twin_axes[i_ch].set_xticklabels(twin_x_ticklabels_final)
        locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
        twin_axes[i_ch].xaxis.set_minor_locator(locmin)

    # Draw 3FGL detection threshold -0.075, 1.075
    for i_xi in range(n_xis):
        for i_ax in range(2):
            rect = mpl.patches.Rectangle((np.log10(4e-10), -0.075), np.log10(5e-10) - np.log10(4e-10), 1.075 + 0.075,
                                         linewidth=0, edgecolor=None, facecolor="#cccccc", zorder=-1)
            axs[i_ax, i_xi].add_patch(rect)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # Save
    if len(save_name) > 0:
        fig.savefig(save_name)
        # plt.close("all")
