from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import os
import sys
plt.ion()
import GCE.gce

gce = GCE.gce.Analysis()
gce.load_params("../parameter_files/realistic_parameters.py")
gce.print_params()

# Ray settings (for parallelized data generation)
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000}
ray_settings = {"num_cpus": 4}  # select the number of CPUs here
gce.generate_template_maps(ray_settings, n_example_plots=5, job_id=0)
gce.combine_template_maps(save_filenames=True, do_combine=True)
gce.build_pipeline()

# import pickle
# with open("../data/Combined_maps/Example_comb_128/Validation/Maps_00_val.pickle", 'rb') as f:
#     data = pickle.load(f)
#
# samples = gce.datasets["val"].get_samples(1)
# data, labels = samples["data"], samples["label"]  # samples contains data and labels (flux fractions & SCD histograms)
# print("Shapes:")
# print("  Data", data.shape)  # n_samples x n_pix_in_ROI
# print("  Flux fractions", labels[0].shape)  # n_samples x n_templates
# print("  SCD histograms", labels[1].shape)  # n_samples x n_bins x n_PS_templates

# NOTE: the maps are stored in NEST format
# map_to_plot = 0
# r = gce.p.data["outer_rad"] + 1
# hp.cartview(gce.decompress(data[map_to_plot] * gce.template_dict["rescale_compressed"]), nest=True,
#             title="Simulated data: Count space", lonra=[-r, r], latra=[-r, r])
# hp.cartview(gce.decompress(data[map_to_plot]), nest=True,
#             title="Simulated data: Flux space", lonra=[-r, r], latra=[-r, r])
# hp.cartview(gce.decompress(gce.template_dict["rescale_compressed"], fill_value=np.nan), nest=True,
#             title="Fermi exposure correction", lonra=[-r, r], latra=[-r, r])
# plt.show()
#
# fermi_counts = gce.datasets["test"].get_fermi_counts()
# hp.cartview(gce.decompress(fermi_counts * gce.generators["test"].settings_dict["rescale_compressed"]), nest=True,
#             title="Fermi data: Count space", max=100, lonra=[-r, r], latra=[-r, r])
# # hp.cartview(gce.decompress(fermi_counts), nest=True, title="Fermi data: Flux space", max=100)
# plt.show()

gce.build_nn()

# gce.load_nn()
# gce.train_nn("flux_fractions")
# gce.train_nn("histograms")

# n_samples = 20
# test_samples = gce.datasets["test"].get_samples(n_samples)
# test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][1]
# tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%
# pred = gce.predict(test_data, tau=tau, multiple_taus=True)  # get the NN predictions
#
# # Make some plots (will be saved in the models folder)
# gce.plot_nn_architecture()
# gce.plot_flux_fractions(test_ffs, pred)
# gce.plot_histograms(test_hists, pred, plot_inds=np.arange(9))
# gce.plot_maps(test_data, decompress=True, plot_inds=np.arange(9))
# plt.show()


# Get Fermi map
fermi_counts = gce.datasets["test"].get_fermi_counts()

# Get the predictions
tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%
fermi_pred = gce.predict(fermi_counts[np.newaxis, :], tau=tau, multiple_taus=True)

import tensorflow as tf
import seaborn as sns
from scipy.integrate import trapz
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, LogLocator
import colorcet as cc
sns.set_style("ticks")
sns.set_context("talk")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 14
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["axes.titlesize"] = 24


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

    fig, axs_hists = plt.subplots(1, 1, figsize=(7.0, 3.1), constrained_layout=True)
    axs_hists.semilogx()

    # For GCE only
    for i_ch in range(1):

        # Iterate over the taus
        for i_tau in range(n_taus):

            # Plot differential histogram
            axs_hists.fill_between(10 ** (bin_centres - width / 2.0), NN_pred[i_tau, :, i_ch], color=colors[i_tau],
                                   zorder=1, alpha=0.075, step="post")

            # For median: plot a solid line
            if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                axs_hists.step(10 ** (bin_centres - width / 2.0), NN_pred[i_tau, :, i_ch], color
                                      ="k", lw=2, zorder=3, alpha=1.0, where="post")

                renorm_const_NN = trapz(NN_pred[:, :, i_ch].mean(0), bin_centres)

                # only median is given: n_eval_points x 2 (GCE, disk) TODO!
                # if len(NPTF_pred.shape) == 2:
                #     renorm_const_NPTF = trapz(NPTF_pred[:, i_ch], NPTF_farray)
                #     pdf_renorm = NPTF_pred[:, i_ch] * renorm_const_NN / renorm_const_NPTF
                #     axs_hists.plot(NPTF_farray, pdf_renorm, color="#800033", lw=2, zorder=5)
                # else:
                #     raise NotImplementedError

        one_ph_flux = 1 / mean_exp

        # Set axes limits
        twin_axes = [None] * 2
        axs_hists.axvline(one_ph_flux, color="orange", ls="--", zorder=4)
        axs_hists.set_ylim([-0.005, 0.1])
        axs_hists.set_xlim([1.67880402e-13, 1e-7])
        axs_hists.set_title("")
        axs_hists.xaxis.set_major_locator(LogLocator(numticks=12, base=10.0))
        axs_hists.xaxis.set_minor_locator(LogLocator(subs=np.linspace(0.1, 0.9, 9), numticks=12, base=10.0))

        # Draw 3FGL detection threshold -0.075, 1.075
        rect = mpl.patches.Rectangle((4e-10, -0.005), 1e-10, 10 ** (0.1+0.005),
                                     linewidth=0, edgecolor=None, facecolor="#ccccccff", zorder=-1)
        axs_hists.add_patch(rect)

        # Twin axis
        def F2S(x):
            return x * mean_exp

        twin_axes = axs_hists.twiny()
        twin_axes.set_xlabel(r"$\bar{S} \ [\mathrm{counts}]$", labelpad=12)
        twin_axes.set_xscale("log")
        twin_axes.set_xlim(F2S(np.asarray(axs_hists.get_xlim())))
        locmaj = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9))
        twin_axes.xaxis.set_major_locator(locmaj)
        locmin = LogLocator(base=10.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
        twin_axes.xaxis.set_minor_locator(locmin)

        axs_hists.set_xlabel(r"$F \ [\mathrm{counts} \, / \, (\mathrm{cm}^2 \ \mathrm{s})]$")
        axs_hists.set_ylabel("SCD")
        plt.tight_layout()

        # Save
        if len(name) > 0:
            fig.savefig(name)
            # plt.close("all")


# Bin settings
log_flux_bins = np.asarray([-np.infty] + list(np.logspace(-12.5, -7.0, 21)) + [np.infty])  # bins for SCD
d_logF = np.diff(np.log10(log_flux_bins))[1]
logF_bin_centers = (np.log10(log_flux_bins)[1:] + np.log10(log_flux_bins)[:-1]) / 2.0
logF_bin_centers[0] = logF_bin_centers[1] - d_logF
logF_bin_centers[-1] = logF_bin_centers[-2] + d_logF

# Make a plot
tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%
make_plot_fermi_NN_and_NPTFit(fermi_pred["hist"][:, 0, :, :].numpy(), None, None, tau, logF_bin_centers, name="", mean_exp=gce.template_dict["mean_exp_roi"])  # TODO!!!