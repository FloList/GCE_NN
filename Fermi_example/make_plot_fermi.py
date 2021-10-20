"""
Plot NN prediction for Fermi data (FFs and SCDs).
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
from matplotlib.ticker import AutoLocator, LogLocator
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


def make_plot_fermi(model, tau_vec, bin_centres, params, test_out, inner_band=2, name="fermi_plot.pdf",
                   mean_exp=0, width=None, residual_clim_fac=4, fermi_counts_tot=None, cbar=False):

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

    total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band, mask_ring=True, inner=0,
                                        outer=params["outer_rad"], nside=params["nside"])
    if params["mask_type_fermi"] == "3FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "3FGL_mask"))).astype(
            bool)
    elif params["mask_type_fermi"] == "4FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "4FGL_mask"))).astype(
            bool)
    total_mask_neg = hp.reorder(total_mask_neg, r2n=True)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    outer_grid = fig.add_gridspec(2, 3, wspace=0, hspace=0)

    axs_hists = [[None] * 2 for _ in range(2)]
    for i in range(2):
        axs_hists[i][0] = fig.add_subplot(outer_grid[i, 1])  # CDF
        axs_hists[i][1] = fig.add_subplot(outer_grid[i, 2])  # PDF
    axs_hists = np.asarray(axs_hists)

    axs_maps = [None] * 3  # map, Poisson model, residual
    inner_grid = outer_grid[:, 0].subgridspec(2, 2, wspace=0, hspace=0)
    axs_maps[0] = fig.add_subplot(inner_grid[0, :])  # Map
    axs_maps[1] = fig.add_subplot(inner_grid[1, 0])  # Poisson model
    axs_maps[2] = fig.add_subplot(inner_grid[1, 1])  # Residual
    axs_maps = np.asarray(axs_maps)

    for ax in axs_maps.flatten():
        empty_ax(ax)

    palette = copy.copy(sns.cm.rocket_r)
    palette.set_bad(color='white', alpha=0)
    palette_res = copy.copy(cc.cm.coolwarm)
    palette_res.set_bad(color='white', alpha=0)

    # First: evaluate NN
    test_out_loc = copy.copy(test_out)

    # Tile for the different quantile levels
    for key in test_out.keys():
        test_out_loc[key] = np.tile(test_out[key][0, :][None], [n_taus] + [1] * (len(test_out[key].shape) - 1))

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

    if fermi_counts_tot is not None:
        plot_data_full = np.round(fermi_counts_tot * plot_data_full)
        plot_data_model = fermi_counts_tot * plot_data_model
        plot_data_res = fermi_counts_tot * plot_data_res

    max_val = max(np.nanmax(plot_data_model), np.nanmax(plot_data_full))
    hp.cartview(plot_data_full, nest=True, title="", cbar=False, max=max_val, min=0)
    fig_, ax_ = plt.gcf(), plt.gca()
    data_ = ax_.get_images()[0].get_array()
    plt.close(fig_)
    im_show_map = axs_maps[0].imshow(data_, cmap=palette, vmin=0, vmax=max_val)
    zoom_ax(axs_maps[0])
    if cbar:
        cbar_data = plt.colorbar(im_show_map, ax=axs_maps[0], orientation="horizontal")

    # Title: FFs ?
    pred_vals = pred_fluxes.mean(0)  # average over tau's - they shouldn't have any effect on FF prediction
    title_map = "Pred: "
    title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in pred_vals])
    FF_STD = np.asarray([np.sqrt(np.diag(pred_fluxes_dict["covar"][_])) for _ in range(n_taus)]).mean(0)
    title_map += "\n" + r"$1\sigma$: "
    title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in FF_STD])
    axs_maps[0].set_title(title_map, size=10)

    # Second: plot Poisson model
    plot_data_model[total_mask_neg] = np.nan
    hp.cartview(plot_data_model, nest=True, title="", cbar=False, max=max_val, min=0)
    fig_, ax_ = plt.gcf(), plt.gca()
    data_ = ax_.get_images()[0].get_array()
    plt.close(fig_)
    im_show_Poisson = axs_maps[1].imshow(data_, cmap=palette, vmin=0, vmax=max_val)
    zoom_ax(axs_maps[1])
    if cbar:
        cbar_Poisson = plt.colorbar(im_show_Poisson, ax=axs_maps[1], orientation="horizontal")

    # Third: plot residual
    max_val_res = max_val / residual_clim_fac
    min_val_res = - max_val / residual_clim_fac
    plot_data_res[total_mask_neg] = np.nan
    hp.cartview(plot_data_res, nest=True, title="", cbar=False, max=max_val_res, min=min_val_res)
    fig_, ax_ = plt.gcf(), plt.gca()
    data_ = ax_.get_images()[0].get_array()
    plt.close(fig_)
    im_show_res = axs_maps[2].imshow(data_, cmap=palette_res, vmin=min_val_res, vmax=max_val_res)
    zoom_ax(axs_maps[2])
    if cbar:
        cbar_res = plt.colorbar(im_show_res, ax=axs_maps[2], orientation="horizontal")

    # For GCE and disk histograms
    for i_ch in range(2):

        # Iterate over the taus
        for i_tau in range(n_taus):

            # Plot differential histogram
            axs_hists[i_ch, 1].fill_between(bin_centres - width / 2.0, pred_hist[i_tau, :, i_ch], color=colors[i_tau],
                                            zorder=1, alpha=0.075, step="post")

            # For median: plot a solid line
            if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                axs_hists[i_ch, 1].step(bin_centres - width / 2.0, pred_hist[i_tau, :, i_ch], color="k", lw=2, zorder=3,
                                        alpha=1.0, where="post")

            # Plot cumulative histogram
            if i_tau < n_taus - 1:
                # Draw the next section of the cumulative histogram in the right colour
                for i in range(len(bin_centres)):
                    # Draw the next section of the cumulative histogram in the right colour
                    axs_hists[i_ch, 0].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                              y1=pred_hist[i_tau, :, i_ch].cumsum()[i],
                                              y2=pred_hist[i_tau + 1, :, i_ch].cumsum()[i], color=colors[i_tau], lw=0)
                    # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                    if i_tau == 0 and pred_hist[0, :, i_ch].cumsum()[i] > 0.99:
                        axs_hists[i_ch, 0].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                          color=colors[0], lw=2, zorder=3)
                    elif i_tau == n_taus - 2 and pred_hist[-1, :, i_ch].cumsum()[i] < 0.01:
                        axs_hists[i_ch, 0].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                          color=colors[-1], lw=2, zorder=3)

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
                                         linewidth=0, edgecolor=None, facecolor="#cccccc", zorder=-1)
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

        if params["which_histogram"] == 1 and i_ch == 1:
            axs_hists[i_ch, 0].set_xlabel(r"$\log_{10} \ F$")  # counts = flux here
            axs_hists[i_ch, 1].set_xlabel(r"$\log_{10} \ F$")
        elif i_ch == 1:
            axs_hists[i_ch, 0].set_xlabel("Count number")
            axs_hists[i_ch, 1].set_xlabel("Count number")

    # Save
    if len(name) > 0:
        filename = os.path.join(model.get_path("checkpoints"), name)
        fig.savefig(filename)
        plt.close("all")
