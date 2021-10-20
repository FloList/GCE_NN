"""
Plot true vs. estimated flux fractions for each template.
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns


def make_error_plot_zoomed(MODELS, zoom_lims, ticks, tick_labels, real_fluxes, pred_fluxes, colours=None, delta_y=0, model_names=None,
                    out_file="error_plot.pdf", legend=None, show_stripes=True, show_stats=True, marker="^",
                    ms=40, alpha=0.4, lw=0.8, text_colours=None, cmap="magma", pred_covar=None, ecolor=None,
                    vmin=None, vmax=None, show_ticks=False):
    """
    Make an error plot of the NN predictions
    :param MODELS: models to plot
    :param zoom_lims: zoom limits for each model
    :param ticks: axes ticks
    :param tick_labels: axes tick labels
    :param real_fluxes: true flux fractions
    :param pred_fluxes: NN estimates of the flux_fractions
    :param colours: colours to use for plotting (default: settings from GCE letter)
    :param delta_y: shift vertical position of the stats
    :param model_names: names of the models, defaults to MODELS
    :param out_file: name of the output file
    :param legend: show legend? by default on if NN and NPTFit fluxes are given, otherwise off
    :param show_stripes: show the stripes indicating 5% and 10% errors?
    :param show_stats: show stats?
    :param marker: marker for NN estimates
    :param ms: marker size for NN estimates
    :param alpha: alpha for the markers
    :param lw: linewidth for the markers
    :param text_colours: defaults to colours
    :param cmap: default: "magma"
    :param pred_covar: predicted covariances for an error bar plot (non-diag. elements are ignored)
    :param ecolor: errorbar colour
    :param vmin / vmax: limits for colourmap
    :param show_ticks: show ticks?
    :return figure and axes
    """

    sns.set_context("talk")
    sns.set_style("ticks" if show_ticks else "white")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    if colours is None:
        colours = ['#ff0000', '#ec7014', '#fec400', '#37c837', 'deepskyblue', 'k']

    if model_names is None:
        model_names = MODELS
    n_col = max(int(np.ceil(np.sqrt(len(MODELS)))), 1)
    n_row = int(np.ceil(len(MODELS) / n_col))

    if text_colours is None and len(colours[0]) == 1:
        text_colours = colours
    elif text_colours is None:
        text_colours = ["k"] * len(MODELS)

    # Calculate errors
    mean_abs_error = np.mean(np.abs(pred_fluxes - real_fluxes), 0)
    max_abs_error = np.max(np.abs(pred_fluxes - real_fluxes), 0)

    scat_fig, scat_ax = plt.subplots(n_row, n_col, figsize=(11.64, 8), squeeze=False, sharex="none", sharey="none")
    for i_ax, ax in enumerate(scat_ax.flatten()):
        if i_ax >= len(MODELS):
            continue
        ax.plot([0, 1], [0, 1], 'k-', lw=2, alpha=0.5)
        if show_stripes:
            ax.fill_between([0, 1], y1=[0.05, 1.05], y2=[-0.05, 0.95], color="0.7", alpha=0.5)
            ax.fill_between([0, 1], y1=[0.1, 1.1], y2=[-0.1, 0.9], color="0.9", alpha=0.5)
        ax.set_aspect("equal", "box")

    for i_ax, ax in enumerate(scat_ax.flatten(), start=0):
        if i_ax >= len(MODELS):
            ax.axis("off")
            continue
        if pred_covar is None:
            ax.scatter(real_fluxes[:, i_ax], pred_fluxes[:, i_ax], s=ms, c=colours[i_ax], marker=marker,
                                 lw=lw, alpha=alpha, edgecolor="k", zorder=3, label="NN", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            if len(np.asarray(colours).shape) > 1:
                ax.errorbar(x=real_fluxes[:, i_ax], y=pred_fluxes[:, i_ax], fmt='none',
                            alpha=alpha, zorder=3, label="",
                            yerr=np.sqrt(pred_covar[:, i_ax, i_ax]), elinewidth=2)
                ax.scatter(real_fluxes[:, i_ax], pred_fluxes[:, i_ax], s=ms, c=colours[i_ax], marker=marker,
                           lw=lw, alpha=alpha, edgecolor="k", zorder=3, label="NN", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.errorbar(x=real_fluxes[:, i_ax], y=pred_fluxes[:, i_ax], fmt=marker, ms=ms, mfc=colours[i_ax], mec="k",
                            ecolor=ecolor or colours[i_ax], lw=lw, alpha=alpha, zorder=3, label="NN", yerr=np.sqrt(pred_covar[:, i_ax, i_ax]),
                            elinewidth=2)
        if i_ax == 0 and legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], frameon=True, loc='upper left', bbox_to_anchor=(0, 0.85),
                      handletextpad=0.07, borderpad=0.25, fontsize=14)

        zoom_len = zoom_lims[i_ax][1] - zoom_lims[i_ax][0]
        zoom_min, zoom_max = zoom_lims[i_ax]

        if show_stats:
            ax.text(zoom_min + 0.64 * zoom_len, zoom_min + 0.14 * zoom_len + delta_y,
                    r"$\it{Mean}$", ha="center", va="center", size=12)
            ax.text(zoom_min + 0.89 * zoom_len, zoom_min + 0.14 * zoom_len + delta_y,
                    r"$\it{Max}$", ha="center", va="center", size=12)
            ax.text(zoom_min + 0.64 * zoom_len, zoom_min + 0.07 * zoom_len + delta_y,
                    "{:.2f}%".format(mean_abs_error[i_ax] * 100), ha="center", va="center",
                    color=text_colours[i_ax], size=12)
            ax.text(zoom_min + 0.89 * zoom_len, zoom_min + 0.07 * zoom_len + delta_y,
                    "{:.2f}%".format(max_abs_error[i_ax] * 100), ha="center", va="center",
                    color=text_colours[i_ax], size=12)

        ax.set_xlim(zoom_lims[i_ax])
        ax.set_ylim(zoom_lims[i_ax])
        ax.set_xticks(ticks[i_ax])
        ax.set_xticklabels(tick_labels[i_ax])
        ax.set_yticks(ticks[i_ax])
        ax.set_yticklabels(tick_labels[i_ax])
        ax.text(zoom_min + 0.05 * zoom_len, zoom_min + 0.95 * zoom_len, model_names[i_ax], va="top", ha="left")

    scat_fig.text(0.5, 0.025, "True", ha="center", va="center", fontsize=14)
    scat_fig.text(0.02, 0.5, "Estimated", ha="center", va="center", rotation="vertical", fontsize=14)
    plt.tight_layout()
    plt.show()

    if out_file is not None:
        scat_fig.savefig(out_file, bbox_inches="tight")

    return scat_fig, scat_ax
