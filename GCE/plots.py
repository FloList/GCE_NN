import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import healpy as hp
import colorcet as cc


def plot_flux_fractions(params, true_ffs, preds, nptfit_ffs=None, out_file="ff_error_plot.pdf", legend=None,
                        show_stripes=True, show_stats=True, delta_y=0, marker="^", marker_nptf="o", ms=8, ms_nptfit=6,
                        alpha=0.4, lw=0.8, lw_nptfit=1.6, ecolor=None, ticks=None, figsize=None):
    """
    Make an error plot of the NN flux fraction predictions.
    :param params: parameter dictionary
    :param true_ffs: true flux fraction labels
    :param preds: NN estimates (output dictionary)
    :param nptfit_ffs: if provided: also plot NPTFit flux fractions
    :param out_file: name of the output file
    :param legend: show legend? by default on if NN and NPTFit fluxes are given, otherwise off
    :param show_stripes: show the stripes indicating 5% and 10% errors?
    :param show_stats: show stats?
    :param delta_y: shift vertical position of the stats
    :param marker: marker for NN estimates
    :param marker_nptf: marker for NPTFit estimates
    :param ms: marker size for NN estimates
    :param ms_nptfit: marker size for NPTFit estimates
    :param alpha: alpha for the markers
    :param lw: linewidth for the markers
    :param lw_nptfit: linewidth for the NPTFit markers
    :param ecolor: error bar color
    :param ticks: ticks
    :param figsize: figure size
    :return figure, axes
    """
    models = params.mod["models"]
    n_models = len(models)
    model_names = params.mod["model_names"]
    colors = params.plot["colors"]

    if not isinstance(preds, dict):
        raise TypeError("Predictions must be a dictionary!")

    if "ff_mean" not in preds.keys():
        raise KeyError("Key 'ff_mean' not found!")
    else:
        pred_ffs = preds["ff_mean"].numpy()

    if "ff_logvar" in preds.keys():
        pred_stds = np.exp(0.5 * preds["ff_logvar"].numpy())
    elif "ff_covar" in preds.keys():
        pred_stds = np.sqrt(np.asarray([np.diag(c) for c in preds["ff_covar"].numpy()]))
    else:
        pred_stds = None

    if legend is None:
        legend = True if nptfit_ffs is not None else False
        
    n_col = max(int(np.ceil(np.sqrt(n_models))), 1)
    n_row = int(np.ceil(n_models / n_col))

    if figsize is None:
        figsize = (n_col * 3.5, n_row * 3.25)
    
    if ticks is None:
        ticks = [0, 0.2, 0.4, 0.6, 0.8]
    x_ticks = y_ticks = ticks

    # Calculate errors
    mean_abs_error = np.mean(np.abs(pred_ffs - true_ffs), 0)
    max_abs_error = np.max(np.abs(pred_ffs - true_ffs), 0)
    # q95_abs_error = np.quantile(np.abs(pred_ffs - true_ffs), .95, axis=0)
    # q99_abs_error = np.quantile(np.abs(pred_ffs - true_ffs), .99, axis=0)

    if nptfit_ffs is not None:
        mean_abs_error_np = np.mean(np.abs(nptfit_ffs - true_ffs), 0)
        max_abs_error_np = np.max(np.abs(nptfit_ffs - true_ffs), 0)
        # q95_abs_error_np = np.quantile(np.abs(nptfit_ffs - true_ffs), .95, axis=0)
        # q99_abs_error_np = np.quantile(np.abs(nptfit_ffs - true_ffs), .99, axis=0)

    scat_fig, scat_ax = plt.subplots(n_row, n_col, figsize=figsize, squeeze=False, sharex="none", sharey="none")
    for i_ax, ax in enumerate(scat_ax.flatten()):
        if i_ax >= len(models):
            continue
        ax.plot([0, 1], [0, 1], 'k-', lw=2, alpha=0.5)
        if show_stripes:
            ax.fill_between([0, 1], y1=[0.05, 1.05], y2=[-0.05, 0.95], color="0.7", alpha=0.5)
            ax.fill_between([0, 1], y1=[0.1, 1.1], y2=[-0.1, 0.9], color="0.9", alpha=0.5)
        ax.set_aspect("equal", "box")

    for i_ax, ax in enumerate(scat_ax.flatten(), start=0):
        if i_ax >= len(models):
            ax.axis("off")
            continue
        if nptfit_ffs is not None:
            ax.scatter(true_ffs[:, i_ax], nptfit_ffs[:, i_ax], s=ms_nptfit**2, c="1.0", marker=marker_nptf,
                       lw=lw_nptfit, alpha=alpha, edgecolor="k", zorder=2, label="NPTFit")
        if pred_stds is None:
            ax.scatter(true_ffs[:, i_ax], pred_ffs[:, i_ax], s=ms**2, c=colors[i_ax], marker=marker,
                       lw=lw, alpha=alpha, edgecolor="k", zorder=3, label="NN")
        else:
            ax.errorbar(x=true_ffs[:, i_ax], y=pred_ffs[:, i_ax], fmt=marker, ms=ms, mfc=colors[i_ax],
                        mec="k", ecolor=ecolor or colors[i_ax], lw=lw, alpha=alpha, zorder=3, label="NN",
                        yerr=pred_stds[:, i_ax], elinewidth=2)
        if i_ax == 0 and legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], frameon=True, loc='upper left', bbox_to_anchor=(0, 0.85),
                      handletextpad=0.07, borderpad=0.25, fontsize=14)
        if show_stats:
            ax.text(0.62, 0.14 + delta_y, r"$\it{Mean}$", ha="center", va="center", size=12)
            ax.text(0.62, 0.07 + delta_y, "{:.2f}%".format(mean_abs_error[i_ax] * 100), ha="center", va="center",
                    color=colors[i_ax], size=12)
            if nptfit_ffs is not None:
                ax.text(0.65, 0.10, "{:.2f}%".format(mean_abs_error_np[i_ax] * 100), ha="center", va="center", size=12)
            ax.text(0.87, 0.14 + delta_y, r"$\it{Max}$", ha="center", va="center", size=12)
            ax.text(0.87, 0.07 + delta_y, "{:.2f}%".format(max_abs_error[i_ax] * 100), ha="center", va="center",
                    color=colors[i_ax], size=12)
            if nptfit_ffs is not None:
                ax.text(0.87, 0.10, "{:.2f}%".format(max_abs_error_np[i_ax] * 100), ha="center", va="center", size=12)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        if ax.get_subplotspec().is_last_row() or (i_ax + n_col >= n_models):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
        else:
            ax.set_xticks([])
        if ax.get_subplotspec().is_first_col():
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks)
        else:
            ax.set_yticks([])
        ax.text(0.03, 0.97, model_names[i_ax], va="top", ha="left")

    scat_fig.text(0.5, 0.025, "True", ha="center", va="center")
    scat_fig.text(0.02, 0.5, "Estimated", ha="center", va="center", rotation="vertical")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.show()

    if out_file is not None:
        save_folder = params.nn["figures_folder"]
        scat_fig.savefig(os.path.join(save_folder, out_file), bbox_inches="tight")

    return scat_fig, scat_ax


def plot_histograms(params, true_hists, preds, pdf=False, out_file="hist_plot.pdf", plot_inds=None,
                    show_one_ph_line=True, mean_exp=None, show_fermi_threshold=None, show_counts_axis=None,
                    eps_draw_line=0.01, true_col=None, xlims=None, ylims=None, figsize=None):
    """
    Plot SCD histograms.
    :param params: parameter dictionary
    :param true_hists: true histogram labels (can be None)
    :param preds: NN estimates (output dictionary)
    :param pdf: if True: plot density histogram instead of cumulative histogram
    :param out_file: name of the output file
    :param plot_inds: indices to plot can be provided (if None: plot for all maps)
    :param show_one_ph_line: if True: show 1 photon line
    :param mean_exp: mean exposure (required for 1 photon line)
    :param show_fermi_threshold: if True: show the Fermi 5 sigma detection threshold (~4 - 5 ph / (cm^2 s))
    :param show_counts_axis: show a second upper x-axis with expected counts
    :param eps_draw_line: for cumulative histogram values in [0, eps_draw_line] or [1 - eps_draw_line, 1], a line is
    drawn for better visibility
    :param true_col: color for true histogram
    :param xlims: x-axis limits
    :param ylims: y-axis limits
    :param figsize: figure size
    :return: list of figures, list of axes
    """
    # Check if histogram labels are given    
    if isinstance(true_hists, np.ndarray):
        has_hist_label = True
    elif true_hists is None:
        has_hist_label = False
    else:
        raise TypeError("Histogram labels have unsupported type {:}.".format(type(true_hists)))

    # Get widths of bins
    bin_centers = params.nn.hist["nn_hist_centers"]
    widths = params.nn.hist["nn_hist_widths"]
    widths_l = params.nn.hist["nn_hist_widths_l"]
    widths_r = params.nn.hist["nn_hist_widths_r"]

    # Get histogram prediction
    pred_hist = preds["hist"].numpy()

    # Earth Mover's pinball loss: take care of quantile levels
    if params.train["hist_loss"].upper() == "EMPL":
        # if there is no quantile dimension (from predict without "multiple_taus"):
        if len(pred_hist.shape) == 3:  # n_maps, n_bins, n_channels
            pred_hist = np.expand_dims(pred_hist, 0)  # add tau axis
            tau_vec_pre = np.unique(np.expand_dims(preds["tau"].numpy().squeeze(1), 0), axis=1)  # unique over map axis
        elif len(pred_hist.shape) == 4:  # n_taus, n_maps, n_bins, n_channels
            tau_vec_pre = np.unique(preds["tau"].numpy().squeeze(2), axis=1)  # unique over map axis
        else:
            raise NotImplementedError

    # If there's no uncertainty quantification: add tau axis anyway
    else:
        if len(pred_hist.shape) == 3:
            pred_hist = np.expand_dims(pred_hist, 0)
        else:
            raise NotImplementedError
        tau_vec_pre = 0.5 * np.ones((1, 1))  # set to 0.5 (median)

    # Now: pred_hist has shape: n_taus x n_maps x n_bins x n_channels
    n_maps = pred_hist.shape[1]

    # Get number of quantile levels
    assert tau_vec_pre.shape[1] == 1, "Different quantile levels for different maps are currently not supported!"
    tau_vec = tau_vec_pre.squeeze(1)  # now, we can squeeze the "maps" axis of tau
    n_taus = len(tau_vec)
    if n_taus > 1:
        colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]
    else:
        colors = ["#ffa18a66"]

    # Indices to plot
    if plot_inds is None:
        plot_inds = np.arange(n_maps)
    else:
        assert max(plot_inds) <= n_maps, "Plot indices outside range: max(plot_inds) = '{:}', but n_maps = {:}".format(
            max(plot_inds), n_maps)
    n_maps_plot = len(plot_inds)

    # Set some plot settings
    if true_col is None:
        true_col = [0.25490196, 0.71372549, 0.76862745, 1]
    true_col_faint = true_col.copy()
    true_col_faint[-1] = 0.2

    if show_counts_axis is None:
        show_counts_axis = mean_exp is not None and params.nn.hist["which_histogram"] == "dNdF"

    if show_fermi_threshold is None:
        show_fermi_threshold = "Fermi" in params.data["exposure"]

    # Axis limits
    if xlims is None:
        xlims = [bin_centers[0] - widths[0], bin_centers[-1] + 2.5 * widths[-1]]
    if ylims is None:
        ylims = [-0.075, 1.075]

    n_col = max(int(np.ceil(np.sqrt(n_maps_plot))), 1)
    n_row = int(np.ceil(n_maps_plot / n_col))
    if figsize is None:
        figsize = (n_col * 4.5, n_row * 3.25)

    eps = eps_draw_line

    all_figs, all_axs = [], []

    # One plot for each histogram template
    for i_ch, ch in enumerate(params.nn.hist["hist_templates"]):

        fig, axs = plt.subplots(n_row, n_col, figsize=figsize, squeeze=False, sharex="none", sharey="none")
        for i_sample, sample in enumerate(plot_inds):

            i_col, i_row = np.unravel_index(i_sample, [n_col, n_row], order="F")  # first l -> r, then t -> b
            ax = axs[i_row, i_col]

            # Iterate over the taus
            for i_tau in range(n_taus):

                # Plot density histogram as a function of the quantile level
                if pdf:
                    # Plot differential histogram
                    ax.fill_between(bin_centers - widths_l, pred_hist[i_tau, sample, :, i_ch], color=colors[i_tau],
                                                    zorder=1, alpha=max(0.05, min(1.0, 1.0 / n_taus)), step="post")

                    # For median: plot a solid line
                    if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                        ax.step(bin_centers - widths_l, pred_hist[i_tau, sample, :, i_ch], color="k", lw=2, zorder=3,
                                alpha=1.0, where="post")

                # Cumulative histogram
                else:
                    # Plot cumulative histogram as a function of the quantile level
                    if n_taus > 1:
                        if i_tau < n_taus - 1:
                            # Draw the next section of the cumulative histogram in the right colour
                            for i in range(len(bin_centers)):
                                # Draw the next section of the cumulative histogram in the right colour
                                ax.fill_between(x=[bin_centers[i] - widths_l[i], bin_centers[i] + widths_r[i]],
                                                               y1=pred_hist[i_tau, sample, :, i_ch].cumsum()[i],
                                                               y2=pred_hist[i_tau + 1, sample, :, i_ch].cumsum()[i],
                                                               color=colors[i_tau], lw=0)
                                # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                                if i_tau == 0 and pred_hist[0, :, i_ch].cumsum()[i] > 1 - eps:
                                    ax.plot([bin_centers[i] - widths_l[i], bin_centers[i] + widths_r[i]],
                                                           2 * [1.0], color=colors[0], lw=2, zorder=3)
                                elif i_tau == n_taus - 2 and pred_hist[-1, sample, :, i_ch].cumsum()[i] < eps:
                                    ax.plot([bin_centers[i] - widths_l[i], bin_centers[i] + widths_r[i]],
                                                           2 * [0.0], color=colors[-1], lw=2, zorder=3)
                    # If a single quantile level is provided: make a bar plot
                    elif n_taus == 1:
                        ax.bar(bin_centers - widths_l, pred_hist[i_tau, sample, :, i_ch].cumsum(), color=colors[i_tau],
                               lw=2, align="edge", width=widths)

            # 1-ph line
            if show_one_ph_line and mean_exp is not None:
                one_ph_flux = 1 / mean_exp
                ax.axvline(one_ph_flux, color="#ff9d00", ls="--")

            # Fermi threshold
            if show_fermi_threshold:
                rect = mpl.patches.Rectangle((4e-10, -0.075), 1e-10, 1.075 + 0.075,
                                             linewidth=0, edgecolor=None, facecolor="#ff9d00", zorder=5)
                ax.add_patch(rect)

            # Bar plot for true histogram
            if has_hist_label:
                true_to_plot = true_hists[sample, :, i_ch] if pdf else true_hists[sample, :, i_ch].cumsum()
                ax.bar(bin_centers - widths_l, true_to_plot, fc=true_col_faint, ec=true_col, width=widths, lw=2,
                       align="edge")

            # Log-spaced?
            if params.nn.hist["log_spaced_bins"]:
                ax.set_xscale("log")

            # Set axes limits
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_title("")

            if not ax.get_subplotspec().is_first_col():
                ax.set_yticks([])
            if not ax.get_subplotspec().is_last_row():
                ax.set_xticks([])

            # Upper x-axis for expected counts?
            if show_counts_axis and ax.get_subplotspec().is_first_row():
                def f2s(x): return x * mean_exp

                # Build twin axes and set limits
                twin_axes = ax.twiny()

                # Set labels and ticks
                twin_axes.set_xscale(ax.get_xscale())
                twin_axes.set_xlim(f2s(np.asarray(xlims)))
                locmaj = LogLocator(base=10.0, numticks=21)
                twin_axes.xaxis.set_major_locator(locmaj)
                locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
                twin_axes.xaxis.set_minor_locator(locmin)

        # Labels
        if params.nn.hist["which_histogram"] == "dNdF":
            xlabel = r"$F \ [\mathrm{counts} \ \mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$"
        else:
            xlabel = "Count number"

        plt.tight_layout()
        plt.subplots_adjust(top=0.845, bottom=0.105, left=0.085, right=0.985, hspace=0.0, wspace=0.0)
        fig.text(0.5, 0.025, xlabel, ha="center", va="center", fontsize=18)
        ylabel_text = "PDF" if pdf else "CDF"
        fig.text(0.02, 0.5, ylabel_text, ha="center", va="center", rotation="vertical", fontsize=18)
        fig.text(0.5, 0.975, params.nn.hist["hist_template_names"][i_ch], ha="center", va="center", fontsize=18)

        if show_counts_axis:
            xlabel_upper = r"$\bar{S}$"
            fig.text(0.5, 0.925, xlabel_upper, ha="center", va="center", fontsize=18)

        # Delete unused axes
        for i_ax, ax in enumerate(axs.flatten(), start=0):
            if i_ax >= n_maps_plot:
                ax.axis("off")

        # Save
        if len(out_file) > 0:
            save_folder = params.nn["figures_folder"]
            out_file_first, out_file_last = out_file.split(".", maxsplit=1)
            pdf_cdf_str = "pdf" if pdf else "cdf"
            fig.savefig(os.path.join(save_folder, out_file_first + "_" + ch + "_" + pdf_cdf_str + "." + out_file_last),
                        bbox_inches="tight")

        all_figs.append(fig)
        all_axs.append(axs)

    return all_figs, all_axs


def plot_maps(maps, params, out_file="maps.pdf", cmap="rocket_r", plot_inds=None, show_tot_counts=True, badcolor="0.5",
              figsize=None):
    """
    Plots the maps.
    :param maps: compressed maps as generated by datasets
    :param params: parameter dictionary
    :param out_file: name of the output file
    :param cmap: colormap for plotting
    :param plot_inds: indices to plot can be provided (if None: plot for all maps)
    :param show_tot_counts: if True: show the total number of counts in the map
    :param badcolor: color for masked pixels
    :param figsize: size of the figure
    :return: figure, axes
    """
    def sep_comma(s): return f'{s:,}'

    # Define the indices for the maps to plot
    n_maps = maps.shape[0]
    if plot_inds is None:
        plot_inds = np.arange(n_maps)
    else:
        assert max(plot_inds) <= n_maps, "Plot indices outside range: max(plot_inds) = '{:}', but n_maps = {:}".format(
            max(plot_inds), n_maps)
    n_maps_plot = len(plot_inds)

    n_col = max(int(np.ceil(np.sqrt(n_maps_plot))), 1)
    n_row = int(np.ceil(n_maps_plot / n_col))

    fig, axs = plt.subplots(n_row, n_col, figsize=figsize, squeeze=False)

    for i_sample, sample in enumerate(plot_inds):
        i_col, i_row = np.unravel_index(i_sample, [n_col, n_row], order="F")  # first l -> r, then t -> b

        # First: plot map
        hp.cartview(maps[sample, :], nest=True, cmap=cmap, badcolor=badcolor, title="", cbar=False, fig=1,
                    sub=(n_row, n_col, i_row * n_col + i_col + 1), max=None, min=0)
        ax = plt.gca()
        ax.set_xlim([-params.data["outer_rad"] - 1, params.data["outer_rad"] + 1])
        ax.set_ylim([-params.data["outer_rad"] - 1, params.data["outer_rad"] + 1])

        if show_tot_counts:
            ax.text(-params.data["outer_rad"], params.data["outer_rad"], sep_comma(int(np.nansum(maps[sample, :]))),
                    va="top", ha="left")
        axs[i_col, i_row].axis("off")

    if len(out_file) > 0:
        save_folder = params.nn["figures_folder"]
        fig.savefig(os.path.join(save_folder, out_file), bbox_inches="tight")
