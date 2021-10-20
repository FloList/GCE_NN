"""
Plot predictions of NN and NPTFit.
"""
import numpy as np
import seaborn as sns
import colorcet as cc
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, LogLocator
from scipy.integrate import trapz


def sep_comma(s):
    return f'{s:,}'


def make_plot_fermi_NN_and_NPTFit(NN_pred, NPTF_pred, NPTF_farray, tau_vec, bin_centres, name="fermi_comparison.pdf",
                   mean_exp=0, width=None):

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

    fig, axs_hists = plt.subplots(2, 2, figsize=(7.6, 6), constrained_layout=True)

    # For GCE and disk histograms
    for i_ch in range(2):

        # Iterate over the taus
        for i_tau in range(n_taus):

            # Plot differential histogram
            axs_hists[i_ch, 1].fill_between(bin_centres - width / 2.0, NN_pred[i_tau, :, i_ch], color=colors[i_tau],
                                            zorder=1, alpha=0.075, step="post")

            # For median: plot a solid line
            if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                axs_hists[i_ch, 1].step(bin_centres - width / 2.0, NN_pred[i_tau, :, i_ch], color="k", lw=2, zorder=3,
                                        alpha=1.0, where="post")

            # Plot cumulative histogram
            if i_tau < n_taus - 1:
                # Draw the next section of the cumulative histogram in the right colour
                for i in range(len(bin_centres)):
                    # Draw the next section of the cumulative histogram in the right colour
                    axs_hists[i_ch, 0].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                              y1=NN_pred[i_tau, :, i_ch].cumsum()[i],
                                              y2=NN_pred[i_tau + 1, :, i_ch].cumsum()[i], color=colors[i_tau], lw=0)
                    # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                    if i_tau == 0 and NN_pred[0, :, i_ch].cumsum()[i] > 0.99:
                        axs_hists[i_ch, 0].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                          color=colors[0], lw=2, zorder=3)
                    elif i_tau == n_taus - 2 and NN_pred[-1, :, i_ch].cumsum()[i] < 0.01:
                        axs_hists[i_ch, 0].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                          color=colors[-1], lw=2, zorder=3)

        renorm_const_NN = trapz(NN_pred[:, :, i_ch].mean(0), bin_centres)

        # only median is given: n_eval_points x 2 (GCE, disk)
        if len(NPTF_pred.shape) == 2:
            axs_hists[i_ch, 0].plot(NPTF_farray, NPTF_pred[:, i_ch].cumsum(), color="#800033", lw=2, zorder=3, alpha=1)
            renorm_const_NPTF = trapz(NPTF_pred[:, i_ch], NPTF_farray)
            pdf_renorm = NPTF_pred[:, i_ch] * renorm_const_NN / renorm_const_NPTF
            axs_hists[i_ch, 1].plot(NPTF_farray, pdf_renorm, color="#800033", lw=2, zorder=5)
            axs_hists[i_ch, 0].fill_between(x=NPTF_farray, y1=NPTF_pred[:, i_ch].cumsum(), color="#800033", lw=2,
                                            zorder=3, alpha=30 / 255.0)
        else:
            raise NotImplementedError

    for i_ch in range(2):
        one_ph_flux = np.log10(1 / mean_exp)

        def F2S(x):
            return 10.0 ** x * mean_exp

        # Set axes limits
        twin_axes = [None] * 2
        for _ in range(2):
            axs_hists[i_ch, _].axvline(one_ph_flux, color="orange", ls="--", zorder=4)
            axs_hists[i_ch, _].set_ylim([-0.075, 1.075])
            axs_hists[i_ch, _].set_ylim([-0.075, 1.075])
            axs_hists[i_ch, _].set_xlim([-13, -8])
            axs_hists[i_ch, _].set_title("")
            # x_ticklabels = [r"$" + str(int(t)) + "$" for t in x_ticks]
            # axs_hists[i_ch, _].set_xticklabels(x_ticklabels)

        for _ in range(2):
            axs_hists[i_ch, _].xaxis.set_major_locator(AutoLocator())

        # Draw 3FGL detection threshold -0.075, 1.075
        for _ in range(2):
            rect = mpl.patches.Rectangle((np.log10(4e-10), -0.075), np.log10(5e-10) - np.log10(4e-10), 1.075 + 0.075,
                                         linewidth=0, edgecolor=None, facecolor="#ccccccff", zorder=-1)
            axs_hists[i_ch, _].add_patch(rect)

    # Build twin axes and set limits
    for _ in range(2):
        twin_axes[_] = axs_hists[0, _].twiny()

    # Set labels and ticks
    for _ in range(2):
        twin_axes[_].set_xlabel(r"$\bar{S}$")
        twin_axes[_].set_xscale("log")
        tick_locs = np.logspace(-2, 4, 7)
        twin_axes[_].set_xticks(F2S(tick_locs))
        locmaj = LogLocator(base=10.0)
        twin_axes[_].xaxis.set_major_locator(locmaj)
        locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
        twin_axes[_].xaxis.set_minor_locator(locmin)

    for _ in range(2):
        twin_axes[_].set_xlim(F2S(np.asarray([-13, -8])))

    for i_ch in range(2):
        if i_ch == 0:
            axs_hists[i_ch, 0].set_xticks([])
            axs_hists[i_ch, 1].set_xticks([])

        axs_hists[i_ch, 1].set_yticks([])
        axs_hists[i_ch, 0].set_xlabel(r"$\log_{10} \ F$")
        axs_hists[i_ch, 1].set_xlabel(r"$\log_{10} \ F$")

    # Save
    if len(name) > 0:
        fig.savefig(name)
        # plt.close("all")
