"""
This script evaluates the NN for the isotropic example on Poissonian maps.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
import os
import copy
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'rocket'
# ########################################################
print("\n\nFound TF", tf.__version__, ".")
tf.compat.v1.disable_eager_execution()

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
    TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_add_two_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    bin_edges = copy.copy(params["gce_hist_bins"])

    out_file = os.path.join(model.get_path("checkpoints"), "poisson_data")

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    n_maps_poiss = 1024
    all_taus = np.linspace(0.05, 0.95, 19)
    n_taus = len(all_taus)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    tau_inds_ebars = [0, 9, 18]

    if not os.path.exists(out_file + ".npz"):
        # Generate isotropic Poissonian maps
        unmasked_pix = params["indexes"][0]
        n_pixels = len(unmasked_pix)
        A_poiss_range = np.asarray([1, 100000]) / n_pixels
        A_poiss = np.random.uniform(*A_poiss_range, size=n_maps_poiss)
        maps_poiss = np.random.poisson(A_poiss[None].T, size=(n_maps_poiss, n_pixels))
        tot_counts = maps_poiss.sum(1)
        print("Min/Mean/Max counts:", tot_counts.min(), tot_counts.mean(), tot_counts.max())

        # Predict
        pred_hist_cum_all = np.empty((n_taus, n_maps_poiss, len(bin_centres)))
        for i_tau, tau in enumerate(all_taus):
            tau_vec = tau * np.ones((n_maps_poiss, 1))
            poiss_pred = model.predict({"data": maps_poiss}, None, tau_hist=tau_vec)
            pred_hist = poiss_pred["gce_hist"][:, :, 0]
            pred_hist_cum_all[i_tau] = pred_hist.cumsum(1)

        np.savez(out_file, pred_hist_cum_all=pred_hist_cum_all, maps_poiss=maps_poiss)
        print("DONE!")

    else:
        data_loaded = np.load(out_file + ".npz")
        pred_hist_cum_all = data_loaded["pred_hist_cum_all"]
        maps_poiss = data_loaded["maps_poiss"]


    # Plot
    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    width = min(np.diff(bin_centres))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Iterate over the taus
    for i_tau in range(n_taus):

        median = np.quantile(pred_hist_cum_all[i_tau], 0.5, axis=0)

        # Plot cumulative histogram
        if i_tau < n_taus - 1:
            median_next = np.quantile(pred_hist_cum_all[i_tau + 1], 0.5, axis=0)

            # Draw the next section of the cumulative histogram in the right colour
            for i in range(len(bin_centres)):
                # Draw the next section of the cumulative histogram in the right colour
                ax.fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                y1=median[i], y2=median_next[i], color=colors[i_tau], lw=0)
                # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                if i_tau == 0 and median[i] > 0.99:
                    ax.plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                           color=colors[0], lw=2, zorder=3)
                elif i_tau == n_taus - 2 and median_next[i] < 0.01:
                    ax.plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                           color=colors[-1], lw=2, zorder=3)

        # For selected indices: plot error bars over the samples
        if i_tau in tau_inds_ebars:
            yerr_low = np.quantile(pred_hist_cum_all[i_tau], 0.5, axis=0) \
                       - np.quantile(pred_hist_cum_all[i_tau], 0.16, axis=0)
            yerr_high = np.quantile(pred_hist_cum_all[i_tau], 0.84, axis=0) \
                        - np.quantile(pred_hist_cum_all[i_tau], 0.5, axis=0)
            yerr = np.vstack([yerr_low, yerr_high])
            plt.errorbar(bin_centres, y=median, ls="none", yerr=yerr, capsize=3, ecolor="#aaa9ad",
                         marker="o", color=colors[i_tau], mec="#aaa9ad", ms=4, markeredgewidth=1, elinewidth=2)

    one_ph_flux = 0
    ax.axvline(one_ph_flux, color="orange", ls="--")

    # Set axes limits
    ax.set_ylim([-0.075, 1.075])
    ax.set_ylim([-0.075, 1.075])
    x_ticks = np.linspace(-1.5, 2.0, 8)
    ax.set_xticks(x_ticks)
    x_ticklabels = [r"$" + str(t) + "$" for t in x_ticks]
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlim([-1.8, 2.3])
    ax.set_title("")
    ax.set_xlabel(r"$\log_{10} \ F$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    fig_name = "poisson_plot.pdf"
    fig.savefig(os.path.join(model.get_path("checkpoints"), fig_name), bbox_inches="tight")

    # # # # # # # # # # # #
    # from make_tile_plot import make_tile_plot
    # # Mixture between Poiss. and PS flux
    # n_samples_PS = n_maps_poiss
    # models_test = params["models"]
    # model_names = params["model_names"]
    # input_test_dict = ds_test.next_element.vars()
    # test_PS = ds_test.get_samples(n_samples_PS)
    # test_data_PS, test_label_PS = test_PS["data"], test_PS["label"]
    # test_gce_hist_PS = test_PS["gce_hist"]
    # real_fluxes = test_label_PS
    #
    # # Add Poissonian maps
    # real_data_PS_and_Poiss = test_data_PS + maps_poiss
    # test_PS["data"] = real_data_PS_and_Poiss
    #
    # mean_exp = generator_test.settings_dict["exp"].mean()
    # band_mask_range = 0
    #
    # n_plot = 9
    # # Get n_plot maps that cover a large range of histograms
    # sort_val = 0.95
    # inds_sorted = np.argsort(np.argmin((test_PS["gce_hist"][:, :, 0].cumsum(1) < sort_val), axis=1))
    # plot_samples = inds_sorted[np.floor(np.linspace(0, n_samples_PS - 1, n_plot)).astype(int)]
    # # re-sort: row/col -> col/row
    # plot_samples = np.reshape(plot_samples, [3, 3]).T.flatten()
    # name_w_poiss = os.path.join(model.get_path("checkpoints"), "tile_plot_w_Poiss.pdf")
    # name_maps_w_poiss = os.path.join(model.get_path("checkpoints"), "tile_plot_maps_w_Poiss.pdf")
    #
    # make_tile_plot(model, all_taus, bin_centres, params, test_PS, plot_samples=plot_samples, inner_band=band_mask_range,
    #                     name=name_w_poiss, name_maps=name_maps_w_poiss, mean_exp=mean_exp)
