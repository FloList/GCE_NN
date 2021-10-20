"""
Plotting function for true vs. estimated SCDs.
Also plots the maps.
"""
import os
import numpy as np
import seaborn as sns
import colorcet as cc
import healpy as hp
import copy
from NPTFit import create_mask as cm
from matplotlib.ticker import LogLocator
from matplotlib import pyplot as plt
from gce_utils import masked_to_full, get_template


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


def make_plot_simulated(model, tau_vec, bin_centres, params, test_out, plot_samples, inner_band=2, name="simulated_plot.pdf",
                        mean_exp=0, width=None, residual_clim_fac=4):

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

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    outer_grid = fig.add_gridspec(n_plot, 3, wspace=0, hspace=0)

    axs_hists = [[None] * 2 for _ in range(n_plot)]
    for i in range(n_plot):
        axs_hists[i][0] = fig.add_subplot(outer_grid[i, 1])  # GCE hist
        axs_hists[i][1] = fig.add_subplot(outer_grid[i, 2])  # Disk hist
    axs_hists = np.asarray(axs_hists)

    axs_maps = [[None] * 3 for _ in range(n_plot)]  # map, Poisson model, residual
    for i in range(n_plot):
        inner_grid = outer_grid[i, 0].subgridspec(2, 2, wspace=0, hspace=0)
        axs_maps[i][0] = fig.add_subplot(inner_grid[0, :])  # Map
        axs_maps[i][1] = fig.add_subplot(inner_grid[1, 0])  # Poisson model
        axs_maps[i][2] = fig.add_subplot(inner_grid[1, 1])  # Residual
    axs_maps = np.asarray(axs_maps)

    for ax in axs_maps.flatten():
        empty_ax(ax)

    palette = copy.copy(sns.cm.rocket_r)
    palette.set_bad(color='white', alpha=0)
    palette_res = copy.copy(cc.cm.coolwarm)
    palette_res.set_bad(color='white', alpha=0)

    for i_sample, db_sample in enumerate(plot_samples):

        # First: evaluate NN
        test_out_loc = copy.copy(test_out)

        # Tile for the different quantile levels
        for key in test_out.keys():
            test_out_loc[key] = np.tile(test_out[key][db_sample, :][None], [n_taus] + [1] * (len(test_out[key].shape) - 1))

        # Predict and get means and histograms
        pred_fluxes_dict = model.predict(test_out_loc, None, tau_hist=np.expand_dims(tau_vec, -1))
        pred_fluxes = pred_fluxes_dict["logits_mean"]
        pred_hist = pred_fluxes_dict["gce_hist"]

        # Get Poisson model and residual
        plot_data_model = copy.copy(masked_to_full(np.mean(np.mean(pred_fluxes_dict["count_maps_modelled_Poiss"], 0), 1),
                                                   params["indexes"][0], nside=params["nside"]))
        plot_data_res = copy.copy(masked_to_full(np.mean(np.mean(pred_fluxes_dict["count_maps_residual"], 0), 1),
                                                 params["indexes"][0], nside=params["nside"]))

        # First: plot map
        # NOTE: take count map saved in pred_fluxes_dict instead of test_out to get NORMALISED map, in terms of COUNTS!
        # need to normalise it to get relative counts!
        plot_data_full = copy.copy(masked_to_full(np.mean(np.mean(pred_fluxes_dict["count_maps"], 0), 1),
                                                  params["indexes"][0], nside=params["nside"]))
        plot_data_full[total_mask_neg] = np.nan

        max_val = max(np.nanmax(plot_data_model), np.nanmax(plot_data_full))
        hp.cartview(plot_data_full, nest=True, title="", cbar=False, max=max_val, min=0)
        fig_, ax_ = plt.gcf(), plt.gca()
        data_ = ax_.get_images()[0].get_array()
        plt.close(fig_)
        axs_maps[i_sample, 0].imshow(data_, cmap=palette, vmin=0, vmax=max_val)
        zoom_ax(axs_maps[i_sample, 0])

        # Title: FFs ?
        pred_vals = pred_fluxes.mean(0)  # average over tau's - they shouldn't have any effect on FF prediction
        title_map = "True: "
        title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in test_out["label"][db_sample]])
        title_map += "\nPred: "
        title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in pred_vals])
        FF_STD = np.asarray([np.sqrt(np.diag(pred_fluxes_dict["covar"][_])) for _ in range(n_taus)]).mean(0)
        title_map += "\n" + r"$1\sigma$: "
        title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in FF_STD])
        axs_maps[i_sample, 0].set_title(title_map, size=6)

        # Second: plot Poisson model
        plot_data_model[total_mask_neg] = np.nan
        hp.cartview(plot_data_model, nest=True, title="", cbar=False, max=max_val, min=0)
        fig_, ax_ = plt.gcf(), plt.gca()
        data_ = ax_.get_images()[0].get_array()
        plt.close(fig_)
        axs_maps[i_sample, 1].imshow(data_, cmap=palette, vmin=0, vmax=max_val)
        zoom_ax(axs_maps[i_sample, 1])

        # Third: plot residual
        max_val_res = max_val / residual_clim_fac
        min_val_res = - max_val / residual_clim_fac
        plot_data_res[total_mask_neg] = np.nan
        hp.cartview(plot_data_res, nest=True, title="", cbar=False, max=max_val_res, min=min_val_res)
        fig_, ax_ = plt.gcf(), plt.gca()
        data_ = ax_.get_images()[0].get_array()
        plt.close(fig_)
        axs_maps[i_sample, 2].imshow(data_, cmap=palette_res, vmin=min_val_res, vmax=max_val_res)
        zoom_ax(axs_maps[i_sample, 2])

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
                        axs_hists[i_sample, i_ch].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                                               y1=pred_hist[i_tau, :, i_ch].cumsum()[i],
                                                               y2=pred_hist[i_tau + 1, :, i_ch].cumsum()[i], color=colors[i_tau], lw=0)
                        # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                        if i_tau == 0 and pred_hist[0, :, i_ch].cumsum()[i] > 0.99:
                            axs_hists[i_sample, i_ch].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                                           color=colors[0], lw=2, zorder=3)
                        elif i_tau == n_taus - 2 and pred_hist[-1, :, i_ch].cumsum()[i] < 0.01:
                            axs_hists[i_sample, i_ch].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                                           color=colors[-1], lw=2, zorder=3)

            one_ph_flux = np.log10(1 / mean_exp)
            axs_hists[i_sample, i_ch].axvline(one_ph_flux, color="orange", ls="--")

            if has_hist_label:
                # Bar plot for true cumulative histogram
                axs_hists[i_sample, i_ch].bar(bin_centres, test_gce_hist[db_sample, :, i_ch].cumsum(), fc=cum_col_faint,
                                              ec=cum_col_1, width=width, lw=2)

            # Set axes limits
            axs_hists[i_sample, i_ch].set_ylim([-0.075, 1.075])
            axs_hists[i_sample, i_ch].set_ylim([-0.075, 1.075])
            x_ticks = np.linspace(-12, -7, 6)
            axs_hists[i_sample, i_ch].set_xticks(x_ticks)
            x_ticklabels = [r"$" + str(int(t)) + "$" for t in x_ticks]
            axs_hists[i_sample, i_ch].set_xticklabels(x_ticklabels)
            axs_hists[i_sample, i_ch].set_xlim([-13, -6])
            axs_hists[i_sample, i_ch].set_title("")

            if i_ch > 0:
                axs_hists[i_sample, i_ch].set_yticks([])
            if i_sample < n_plot - 1:
                axs_hists[i_sample, i_ch].set_xticks([])

            if params["which_histogram"] == 1 and i_sample == n_plot - 1:
                axs_hists[i_sample, i_ch].set_xlabel(r"$\log_{10} \ F$")  # counts = flux here
                axs_hists[i_sample, i_ch].set_xlabel(r"$\log_{10} \ F$")
            elif i_sample == n_plot - 1:
                axs_hists[i_sample, i_ch].set_xlabel("Count number")
                axs_hists[i_sample, i_ch].set_xlabel("Count number")

    # Upper x axis with expected counts
    for i_ch in range(2):
        twin_axes[i_ch] = axs_hists[0, i_ch].twiny()

    for i_ch in range(2):
        twin_axes[i_ch].set_xlabel(r"$\bar{S}$")
        twin_axes[i_ch].set_xscale("log")
        twin_axes[i_ch].set_xlim(F2S(np.asarray([-13, -6])))

    for i_ch in range(2):
        tick_locs_logS = np.logspace(-2, 4, 7)
        twin_axes[i_ch].set_xticks(tick_locs_logS)
        locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
        twin_axes[i_ch].xaxis.set_minor_locator(locmin)

    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # Save
    if len(name) > 0:
        filename = os.path.join(model.get_path("checkpoints"), name)
        fig.savefig(filename)
        # plt.close("all")
