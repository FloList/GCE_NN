"""
Plotting function for the mismodelling experiment.
"""
import numpy as np
import scipy as sp
import seaborn as sns
import colorcet as cc
import matplotlib as mpl
from matplotlib.ticker import LogLocator
from matplotlib import pyplot as plt


def make_plot_mismodelling(pred_FF_all, pred_std_all, pred_hist_all, tau_vec, bin_centres, params,
                           filename="simulated_plot_mismodelling.pdf", mean_exp=0, width=None, cum=False, colours=None,
                           true_FFs=None, true_hists=None, true_ax_ind=None, exclude_FF_inds=[]):

    # Define colours for models
    if colours is None:
        colours = ['#ff0000', '#ec7014', '#fec44f', '#37c837', 'deepskyblue', 'k']

    # Set some plot settings
    # col_truth = "#800033"
    col_truth = "#ff74a5"

    # Get width of bins
    if width is None:
        width = min(np.diff(bin_centres))

    # Get number of quantile levels
    n_taus = len(tau_vec)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]
    assert n_taus == pred_hist_all.shape[1], "Number of quantile levels tau does not match prediction!"

    n_plot, n_models = pred_FF_all.shape

    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    n_rows = n_plot
    n_cols = 3  # FFs, GCE hist, disk hist

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows), constrained_layout=True)

    for i_plot in range(n_plot):

        pred_hist = pred_hist_all[i_plot]
        pred_FF = pred_FF_all[i_plot]
        pred_std = pred_std_all[i_plot]
        twin_axes = [None] * 2

        def F2S(x):
            return 10.0 ** x * mean_exp

        # For GCE and disk histograms
        for i_ch in range(2):

            # Plot truth:
            if (i_plot == true_ax_ind or true_ax_ind is None) and true_hists is not None:
                if cum:
                    axs[i_plot, i_ch + 1].step(bin_centres - width / 2.0, true_hists[:, i_ch].cumsum(), color=col_truth,
                                               lw=1, zorder=4, alpha=1.0, where="post", ls="--")
                else:
                    axs[i_plot, i_ch + 1].step(bin_centres - width / 2.0, true_hists[:, i_ch], color=col_truth,
                                               lw=1, zorder=4, alpha=1.0, where="post", ls="--")

            # Iterate over the taus
            for i_tau in range(n_taus):

                if cum:
                    # Plot cumulative histogram
                    if i_tau < n_taus - 1:
                        # Draw the next section of the cumulative histogram in the right colour
                        for i in range(len(bin_centres)):
                            # Draw the next section of the cumulative histogram in the right colour
                            axs[i_plot, i_ch + 1].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                                               y1=pred_hist[i_tau, :, i_ch].cumsum()[i],
                                                               y2=pred_hist[i_tau + 1, :, i_ch].cumsum()[i], color=colors[i_tau], lw=0)
                            # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                            if i_tau == 0 and pred_hist[0, :, i_ch].cumsum()[i] > 0.99:
                                axs[i_plot, i_ch + 1].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                                           color=colors[0], lw=2, zorder=3)
                            elif i_tau == n_taus - 2 and pred_hist[-1, :, i_ch].cumsum()[i] < 0.01:
                                axs[i_plot, i_ch + 1].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                                           color=colors[-1], lw=2, zorder=3)

                else:
                    # Plot differential histogram
                    axs[i_plot, i_ch + 1].fill_between(bin_centres - width / 2.0, pred_hist[i_tau, :, i_ch],
                                                       color=colors[i_tau], zorder=1, alpha=0.075, step="post")

                    # For median: plot a solid line
                    if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                        axs[i_plot, i_ch + 1].step(bin_centres - width / 2.0, pred_hist[i_tau, :, i_ch], color="k",
                                                   lw=1.5, zorder=3, alpha=1.0, where="post")

            one_ph_flux = np.log10(1 / mean_exp)
            axs[i_plot, i_ch + 1].axvline(one_ph_flux, color="orange", ls="--")

        # Plot FFs
        std_fac = 4
        lw = 1.5
        do_fill = True
        for i_model in range(n_models):
            y_vec = np.linspace(pred_FF[i_model] - std_fac * pred_std[i_model],
                                pred_FF[i_model] + std_fac * pred_std[i_model], 1000)
            if do_fill:
                axs[i_plot, 0].fill_between(100 * y_vec, sp.stats.norm.pdf(y_vec, pred_FF[i_model], pred_std[i_model]),
                                            color=colours[i_model],
                                            lw=lw, linestyle="-", alpha=0.175)
                axs[i_plot, 0].plot(100 * y_vec, sp.stats.norm.pdf(y_vec, pred_FF[i_model], pred_std[i_model]),
                                    color=colours[i_model], lw=lw,
                                    linestyle="-", label=str(i_model))
            else:
                axs[i_plot, 0].plot(100 * y_vec, sp.stats.norm.pdf(y_vec, pred_FF[i_model], pred_std[i_model]),
                                    color=colours[i_model], lw=lw, linestyle="-", label=str(i_model))

            # Plot truth:
            if (i_plot == true_ax_ind or true_ax_ind is None) and true_FFs is not None and i_plot not in exclude_FF_inds:
                axs[i_plot, 0].axvline(100 * true_FFs[i_model], color=colours[i_model], lw=lw, ls="--")

        axs[i_plot, 0].set_xlim([0, 65])
        axs[i_plot, 0].set_ylim([0, 120])
        xticks = np.arange(0, 70, 10)
        axs[i_plot, 0].set_xticks(xticks)
        if i_plot == n_plot - 1:
            axs[i_plot, 0].set_xlabel(r"Flux fractions [$\%$]")
        else:
            axs[i_plot, 0].set_xticklabels([])
        if i_plot == n_plot // 2:
            axs[i_plot, 0].set_ylabel("Probability density")

    # Adjust plots
    for i_plot in range(n_plot):
        for i_ch in range(1, 3):
            # Set axes limits
            if cum:
                axs[i_plot, i_ch].set_ylim([-0.075, 1.075])
                y_ticks = np.linspace(0, 1, 6)
                axs[i_plot, i_ch].set_yticks(y_ticks)
            else:
                axs[i_plot, i_ch].set_ylim([-0.075, 0.79])
                y_ticks = np.linspace(0, 0.6, 4)
                axs[i_plot, i_ch].set_yticks(y_ticks)
            x_ticks = np.linspace(-13, -8, 6)
            axs[i_plot, i_ch].set_xticks(x_ticks)
            x_ticklabels = [r"$" + str(int(t)) + "$" for t in x_ticks]
            axs[i_plot, i_ch].set_xticklabels(x_ticklabels)
            axs[i_plot, i_ch].set_xlim([-13, -8])
            axs[i_plot, i_ch].set_title("")

            if i_ch == 2:
                axs[i_plot, i_ch].set_yticks([])
            if i_plot < n_plot - 1:
                axs[i_plot, i_ch].set_xticklabels([])

            if params["which_histogram"] == 1 and i_plot == n_plot - 1 and i_ch == 1:
                axs[i_plot, i_ch].set_xlabel(r"$\log_{10} \ F$")
            elif i_plot == n_plot - 1 and i_ch == 1:
                axs[i_plot, i_ch].set_xlabel("Count number")
            else:
                axs[i_plot, i_ch].set_xlabel("")

            # Draw 3FGL detection threshold
            rect = mpl.patches.Rectangle((np.log10(4e-10), -0.075), np.log10(5e-10) - np.log10(4e-10), 1.075 + 0.075,
                                         linewidth=0, edgecolor=None, facecolor="#cccccc", zorder=-1)
            axs[i_plot, i_ch].add_patch(rect)

    # Upper x axis with expected counts
    for i_ch in range(2):
        for i_plot in range(1):
            twin_axes[i_ch] = axs[i_plot, i_ch + 1].twiny()
            twin_axes[i_ch].set_xscale("log")
            twin_axes[i_ch].set_xlim(F2S(np.asarray([-13, -8])))
            tick_locs_logS = np.logspace(-2, 3, 6)
            twin_axes[i_ch].set_xticks(tick_locs_logS)
            locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
            twin_axes[i_ch].xaxis.set_minor_locator(locmin)
            if i_ch == 0:
                twin_axes[i_ch].set_xlabel(r"$\bar{S}$")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.2)

    # Save
    if len(filename) > 0:
        fig.savefig(filename, bbox_inches="tight")
        # plt.close("all")
