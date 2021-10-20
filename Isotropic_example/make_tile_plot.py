"""
This file contains the function for the creation of the true vs. estimated SCD histogram plots in the isotropic example.
"""

import os
import numpy as np
import seaborn as sns
import colorcet as cc
import healpy as hp
import copy
from NPTFit import create_mask as cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from gce_utils import masked_to_full, get_template


def sep_comma(s):
    return f'{s:,}'


def make_tile_plot(model, tau_vec, bin_centres, params, test_out, plot_samples, inner_band=2,
                   name="tile_plot.pdf", name_maps="tile_plot_maps.pdf", mean_exp=0, width=None, cmap_hp="rocket_r"):

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

    total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band, mask_ring=True, inner=0,
                                        outer=params["outer_rad"], nside=params["nside"])
    if params["mask_type_fermi"] == "3FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "3FGL_mask"))).astype(
            bool)
    elif params["mask_type_fermi"] == "4FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "4FGL_mask"))).astype(
            bool)
    total_mask_neg = hp.reorder(total_mask_neg, r2n=True)


    assert np.abs(np.sqrt(n_plot) % 1 - 0 < 1e-5), "n_plot needs to be a square number!"
    nd = int(np.sqrt(n_plot))
    fig, axs = plt.subplots(nd, nd, figsize=(12, 12), squeeze=False)
    fig_map, axs_map = plt.subplots(nd, nd, figsize=(12, 12), squeeze=False)

    for i_sample, db_sample in enumerate(plot_samples):

        i_col, i_row = np.unravel_index(i_sample, [nd, nd])

        # First: plot map
        plot_data_full = copy.copy(masked_to_full(test_out["data"][db_sample, :], params["indexes"][0], nside=params["nside"]))
        plot_data_full[total_mask_neg] = np.nan
        hp.cartview(plot_data_full, nest=True, cmap=cmap_hp, badcolor="white", title="", cbar=False, fig=1,
                        sub=(nd, nd, i_row * nd + i_col + 1), max=None, min=0)
        ax = plt.gca()
        ax.set_xlim([-params["outer_rad"] - 1, params["outer_rad"] + 1])
        ax.set_ylim([-params["outer_rad"] - 1, params["outer_rad"] + 1])
        ax.text(-params["outer_rad"], params["outer_rad"], sep_comma(int(test_out["data"][db_sample, :].sum())),
                        va="top", ha="left")
        axs_map[i_col, i_row].axis("off")

        # Now: prediction and histogram plots
        test_out_loc = copy.copy(test_out)

        # Tile for the different quantile levels
        for key in test_out.keys():
            test_out_loc[key] = np.tile(test_out[key][db_sample, :][None],
                                        [n_taus] + [1] * (len(test_out[key].shape) - 1))

        # Predict and get means and histograms
        pred_fluxes_dict = model.predict(test_out_loc, None, tau_hist=np.expand_dims(tau_vec, -1))
        pred_hist = pred_fluxes_dict["gce_hist"]

        i_ch = 0  # only 1 channel here

        # Iterate over the taus
        for i_tau in range(n_taus):

            # Plot cumulative histogram
            if i_tau < n_taus - 1:
                # Draw the next section of the cumulative histogram in the right colour
                for i in range(len(bin_centres)):
                    # Draw the next section of the cumulative histogram in the right colour
                    axs[i_row, i_col].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                              y1=pred_hist[i_tau, :, i_ch].cumsum()[i],
                                              y2=pred_hist[i_tau + 1, :, i_ch].cumsum()[i], color=colors[i_tau],
                                              lw=0)
                    # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                    if i_tau == 0 and pred_hist[0, :, i_ch].cumsum()[i] > 0.99:
                        axs[i_row, i_col].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                          2 * [1.0],
                                          color=colors[0], lw=2, zorder=3)
                    elif i_tau == n_taus - 2 and pred_hist[-1, :, i_ch].cumsum()[i] < 0.01:
                        axs[i_row, i_col].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                          2 * [0.0],
                                          color=colors[-1], lw=2, zorder=3)

        one_ph_flux = np.log10(1 / mean_exp)
        axs[i_row, i_col].axvline(one_ph_flux, color="orange", ls="--")

        if has_hist_label:
            # Bar plot for true cumulative histogram
            axs[i_row, i_col].bar(bin_centres, test_gce_hist[db_sample, :, i_ch].cumsum(), fc=cum_col_faint,
                             ec=cum_col_1, width=width, lw=2)

        # Set axes limits
        axs[i_row, i_col].set_ylim([-0.075, 1.075])
        axs[i_row, i_col].set_ylim([-0.075, 1.075])
        x_ticks = np.linspace(-1.5, 2.0, 8)
        axs[i_row, i_col].set_xticks(x_ticks)
        x_ticklabels = [r"$" + str(t) + "$" for t in x_ticks]
        axs[i_row, i_col].set_xticklabels(x_ticklabels)
        axs[i_row, i_col].set_xlim([-1.8, 2.3])
        axs[i_row, i_col].set_title("")

        if i_col > 0:
            axs[i_row, i_col].set_yticks([])
        if i_row < nd - 1:
            axs[i_row, i_col].set_xticks([])

        if params["which_histogram"] == 1 and i_row == nd - 1:
            axs[i_row, i_col].set_xlabel(r"$\log_{10} \ F$")  # counts = flux here
            axs[i_row, i_col].set_xlabel(r"$\log_{10} \ F$")
        elif i_row == nd - 1:
            axs[i_row, i_col].set_xlabel("Count number")
            axs[i_row, i_col].set_xlabel("Count number")

    for i in range(2):
        plt.figure(i)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # Save
    if len(name) > 0:
        filename = os.path.join(model.get_path("checkpoints"), name)
        fig.savefig(filename)
        filename_maps = os.path.join(model.get_path("checkpoints"), name_maps)
        fig_map.savefig(filename_maps)
        plt.close("all")
