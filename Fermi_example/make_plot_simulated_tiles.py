"""
Plotting function for true vs. estimated SCDs.
Different layout as compared to make_plot_simulated.
"""
import os
import numpy as np
import seaborn as sns
import colorcet as cc
import copy
from matplotlib.ticker import LogLocator
from matplotlib import pyplot as plt


def sep_comma(s):
    return f'{s:,}'


def empty_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def zoom_ax(ax):
    ax.set_xlim([340, 460])
    ax.set_ylim([140, 260])


def make_plot_simulated_tiles(model, tau_vec, bin_centres, params, test_out, plot_samples, layout, hist_inds,
                              name="simulated_plot_tiles.pdf", mean_exp=0, width=None, pred_hist_all=None):

    # Check if histogram label is given
    if "gce_hist" in test_out.keys():
        test_gce_hist = test_out["gce_hist"]
        has_hist_label = True
    else:
        has_hist_label = False

    # Get width of bins
    if width is None:
        width = min(np.diff(bin_centres))

    # Get number of quantile levels
    n_taus = len(tau_vec)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    # Define the indices for the maps to plot
    if isinstance(plot_samples, int):
        plot_samples = np.arange(plot_samples)
    n_plot = len(plot_samples)

    # Set some plot settings
    cum_col_1 = [0.25490196, 0.71372549, 0.76862745, 1]
    cum_col_faint = [0.25490196, 0.71372549, 0.76862745, 0.2]

    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    assert n_plot == np.product(layout), "Layout doesn't match number of plot samples!"
    n_rows, n_cols = layout

    fig, axs = plt.subplots(2 * n_rows + 1, n_cols, figsize=(12, 13), constrained_layout=True,
                            gridspec_kw={"height_ratios": n_rows*[1] + [0.4] + n_rows*[1]})
    [axs[n_rows, _].remove() for _ in range(n_cols)]  # Remove dummy axes with space
    axs = np.concatenate([axs[:n_rows, :], axs[n_rows+1:, :]], 0)
    axs = axs.reshape([2, -1])
    axs = axs.T

    for i_sample, db_sample in enumerate(plot_samples):

        # First: evaluate NN
        test_out_loc = copy.copy(test_out)

        # Tile for the different quantile levels
        for key in test_out.keys():
            test_out_loc[key] = np.tile(test_out[key][db_sample, :][None], [n_taus] + [1] * (len(test_out[key].shape) - 1))

        # Predict and get means and histograms
        if pred_hist_all is None:
            pred_fluxes_dict = model.predict(test_out_loc, None, tau_hist=np.expand_dims(tau_vec, -1))
            pred_hist = pred_fluxes_dict["gce_hist"]
        else:
            pred_hist = pred_hist_all[i_sample]

        twin_axes = [None] * 2

        def F2S(x):
            return 10.0 ** x * mean_exp

        # For GCE and disk histograms
        for i_ch in range(2):

            # Iterate over the taus
            for i_tau in range(n_taus):

                # Plot cumulative histogram
                if i_tau < n_taus - 1:
                    # Draw the next section of the cumulative histogram in the right colour
                    for i in range(len(bin_centres)):
                        # Draw the next section of the cumulative histogram in the right colour
                        axs[i_sample, i_ch].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                                         y1=pred_hist[i_tau, :, i_ch].cumsum()[i],
                                                         y2=pred_hist[i_tau + 1, :, i_ch].cumsum()[i], color=colors[i_tau], lw=0)
                        # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                        if i_tau == 0 and pred_hist[0, :, i_ch].cumsum()[i] > 0.99:
                            axs[i_sample, i_ch].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                                     color=colors[0], lw=2, zorder=3)
                        elif i_tau == n_taus - 2 and pred_hist[-1, :, i_ch].cumsum()[i] < 0.01:
                            axs[i_sample, i_ch].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                                     color=colors[-1], lw=2, zorder=3)

            one_ph_flux = np.log10(1 / mean_exp)
            axs[i_sample, i_ch].axvline(one_ph_flux, color="orange", ls="--")

            if has_hist_label:
                # Bar plot for true cumulative histogram
                axs[i_sample, i_ch].bar(bin_centres, test_gce_hist[db_sample, :, i_ch].cumsum(), fc=cum_col_faint,
                                        ec=cum_col_1, width=width, lw=2)

            # Print true FF
            FF = "{:.1f}".format(100 * test_out["label"][db_sample, hist_inds[i_ch]])
            FF += "%"
            props = dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='k', lw=2)
            axs[i_sample, i_ch].text(-8, 0.05, FF, bbox=props)

    # Adjust plot
    for i_sample, db_sample in enumerate(plot_samples):
        row, col = np.unravel_index(i_sample, [2 * n_rows, n_cols])

        for i_ch in range(2):
            # Set axes limits
            axs[i_sample, i_ch].set_ylim([-0.075, 1.075])
            axs[i_sample, i_ch].set_ylim([-0.075, 1.075])
            x_ticks = np.linspace(-12, -7, 6)
            axs[i_sample, i_ch].set_xticks(x_ticks)
            y_ticks = np.linspace(0, 1, 6)
            axs[i_sample, i_ch].set_yticks(y_ticks)
            x_ticklabels = [r"$" + str(int(t)) + "$" for t in x_ticks]
            axs[i_sample, i_ch].set_xticklabels(x_ticklabels)
            axs[i_sample, i_ch].set_xlim([-13, -6])
            axs[i_sample, i_ch].set_title("")

            if not col == 0:
                axs[i_sample, i_ch].set_yticks([])
            if not row == n_rows - 1:
                axs[i_sample, i_ch].set_xticks([])

            if params["which_histogram"] == 1 and row == n_rows - 1 and i_ch == 1:
                axs[i_sample, i_ch].set_xlabel(r"$\log_{10} \ F$")
            elif row == n_rows - 1 and i_ch == 1:
                axs[i_sample, i_ch].set_xlabel("Count number")
            else:
                axs[i_sample, i_ch].set_xlabel("")

    # Upper x axis with expected counts
    for i_ch in range(2):
        for i in range(n_cols):
            twin_axes[i_ch] = axs[i, i_ch].twiny()
            twin_axes[i_ch].set_xlabel(r"$\bar{S}$")
            twin_axes[i_ch].set_xscale("log")
            twin_axes[i_ch].set_xlim(F2S(np.asarray([-13, -6])))
            tick_locs_logS = np.logspace(-2, 4, 7)
            twin_axes[i_ch].set_xticks(tick_locs_logS)
            locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
            twin_axes[i_ch].xaxis.set_minor_locator(locmin)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # Save
    if len(name) > 0:
        filename = os.path.join(model.get_path("checkpoints"), name)
        fig.savefig(filename)
        # plt.close("all")
