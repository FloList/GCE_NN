"""
Script for GCE DM flux artificially injected into the Fermi map.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from make_plot_injection import make_plot_injection
from deepsphere_GCE_workflow import build_NN
import os
import copy
import seaborn as sns
sns.set_style("ticks")
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
    TEST_EXP_PATH = "./checkpoints/Fermi_example_add_two_256_BN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    # TEST_EXP_PATH = "./checkpoints/Fermi_example_add_two_BN_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    bin_edges = copy.copy(params["gce_hist_bins"])
    data_out_file = "Fermi_injection_data"
    nside = params["nside"]

    fermi_folder = get_fermi_folder_basename(GADI, w573=True)
    fermi_folder += "/fermi_data_" + str(nside)

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Get Fermi map prediction
    try:
        fermi_pred_data = np.load(os.path.join(model.get_path("checkpoints"), "fermi_prediction.npz"),
                                  allow_pickle=True)
        print(fermi_pred_data.files)
        fermi_pred = fermi_pred_data["fermi_pred"][()]
        bin_centres = fermi_pred_data["bin_centres"][()]
        tau_vec = fermi_pred_data["tau_vec"][()]
    except FileNotFoundError:
        raise FileNotFoundError("Run the script 'save_fermi_prediction' first!")

    # Get FFs and histogram
    total_fermi_counts = (generator_test.settings_dict["rescale"] * fermi_counts).sum()  # total counts in ROI
    exp_compressed_nest = generator_test.settings_dict["exp"][generator_test.settings_dict["unmasked_pix"]]
    total_fermi_flux = (generator_test.settings_dict["rescale"] * fermi_counts / exp_compressed_nest).sum()
    FFs = fermi_pred["logits_mean"].mean(0)  # avg. over taus (identical)
    fluxes = FFs * total_fermi_flux

    # We have:
    #     f^new_t   = f^old_t     + Delta f_t
    #     f^new_tot = f^old_tot   + Delta f_t
    #     Delta f_t = xi * f^new_tot
    #     Delta f_t = xi / (1 - xi) * f^old_tot

    n_maps = 64
    xi_vec = np.linspace(0, 0.08, 9)[1:]
    n_xi = len(xi_vec)

    np.random.seed(0)
    tf.random.set_seed(0)
    T_ring = get_template(fermi_folder, "gce_12")
    T = hp.reorder(T_ring, r2n=True)  # ring -> nest

    # Set up the mask for the ROI
    total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=2, mask_ring=True, inner=0,
                                        outer=params["outer_rad"], nside=nside)

    MASK_TYPE = params["mask_type_fermi"]
    if MASK_TYPE == "3FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_folder, "3FGL_mask"))).astype(bool)
    elif MASK_TYPE == "4FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_folder, "4FGL_mask"))).astype(bool)

    total_mask_neg = hp.reorder(total_mask_neg, r2n=True)  # ring -> nest

    # Get pixels that are not masked
    unmasked_pix = np.argwhere(1 - total_mask_neg).flatten()
    assert np.all(unmasked_pix == generator_test.settings_dict["unmasked_pix"]), "Something's wrong with the mask!"

    # Mask template and compress
    T_masked_compressed = (T * (1 - total_mask_neg))[unmasked_pix]

    # Define quantile levels tau
    tau_vec = np.linspace(0.05, 0.95, 19)
    # tau_vec = np.asarray([0.25, 0.5, 0.75])
    n_taus = len(tau_vec)

    all_FFs_inj = np.empty((n_xi, n_taus, n_maps, len(FFs)))
    all_FF_stds_inj = np.empty((n_xi, n_taus, n_maps, len(FFs)))
    all_hists_inj = np.empty((n_xi, n_taus, n_maps, len(bin_centres), 2))  # last dim: GCE and disk

    if not os.path.exists(os.path.join(model.get_path("checkpoints"), data_out_file + ".npz")):

        for i_xi, xi in enumerate(xi_vec):

            # Define the amplitude that leads to an expected FF of xi
            xi_wrt_old = xi / (1 - xi)
            temp_sum_flux = (T_masked_compressed / exp_compressed_nest).sum()
            A = xi_wrt_old * total_fermi_flux / temp_sum_flux

            # Generate the maps
            # EXPOSURE IS ALREADY INCORPORATED IN TEMPLATES!
            sim_maps = np.random.poisson(A * T_masked_compressed[None], size=(n_maps, len(T_masked_compressed)))

            mean_counts_tot = sim_maps.sum(1).mean()
            count_frac_pre = mean_counts_tot / total_fermi_counts
            count_frac_post = mean_counts_tot / (mean_counts_tot + total_fermi_counts)
            print("  xi:", "{:2.2f}".format(xi), "  mean counts:", mean_counts_tot, "=", "{:2.2f}".format(100 * count_frac_pre),
                  "% of Fermi pre =", "{:2.2f}".format(100 * count_frac_post), "% of Fermi post")

            # Remove exposure correction
            if params["remove_exp"]:
                sim_maps_final = sim_maps / generator_test.settings_dict["rescale"][None]
            else:
                sim_maps_final = sim_maps.copy()

            fermi_maps_inj = fermi_counts + sim_maps_final

            # Predict
            for i_tau, tau in enumerate(tau_vec):
                this_pred = model.predict({"data": fermi_maps_inj}, None, tau_hist=tau * np.ones((n_maps, 1)))
                all_FFs_inj[i_xi, i_tau, :, :] = this_pred["logits_mean"]
                all_FF_stds_inj[i_xi, i_tau, :, :] = np.sqrt([np.diag(this_pred["covar"][_]) for _ in range(n_maps)])
                all_hists_inj[i_xi, i_tau, :, :, :] = this_pred["gce_hist"]

        # For FFs and FF stds: average over tau dimension (independent of tau)
        all_FFs_inj = all_FFs_inj.mean(1)
        all_FF_stds_inj = all_FF_stds_inj.mean(1)

        # Save:
        np.savez(os.path.join(model.get_path("checkpoints"), data_out_file), all_FFs_inj=all_FFs_inj,
                 all_FF_stds_inj=all_FF_stds_inj, all_hists_inj=all_hists_inj, tau_vec=tau_vec, xi_vec=xi_vec)
        print("Predictions saved!")

    else:
        inj_data = np.load(os.path.join(model.get_path("checkpoints"), data_out_file + ".npz"), allow_pickle=True)
        all_FFs_inj = inj_data["all_FFs_inj"][()]
        all_FF_stds_inj = inj_data["all_FF_stds_inj"][()]
        all_hists_inj = inj_data["all_hists_inj"][()]

    # 1) Make a plot of the injected vs recovered GCE flux fractions
    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    fig_ff, ax_ff = plt.subplots(1, 1, figsize=(4, 4))
    ax = ax_ff
    GCE_ind = 4
    FFs_inj_GCE = all_FFs_inj[:, :, GCE_ind]
    # Line for truth:
    # at xi = 0:  (xi, FF) = (0, FP), where FP = Fermi prediction (without injection)
    # at max_xi:  (xi, FF) = (max_xi, (GCE flux orig + Delta flux) / (Total flux orig + Delta flux))
    max_xi = 0.08
    GCE_flux_orig = FFs[GCE_ind] * total_fermi_flux
    Delta_f_xi_max = total_fermi_flux * max_xi / (1 - max_xi)
    exp_FF_xi_max = (GCE_flux_orig + Delta_f_xi_max) / (total_fermi_flux + Delta_f_xi_max)

    ax.plot([0, 100 * max_xi], [100 * FFs[GCE_ind], 100 * exp_FF_xi_max], "k--", lw=1, alpha=0.4)

    # Prediction for Fermi map
    ax.plot(0, 100 * FFs[GCE_ind], ls="none", marker="o", ms=5, color="k", mfc="white", mew=2)

    # Predictions for injection maps
    FF_median = np.median(FFs_inj_GCE, axis=1)
    yerr_low = np.quantile(FFs_inj_GCE, 0.5, axis=1) - np.quantile(FFs_inj_GCE, 0.16, axis=1)
    yerr_high = np.quantile(FFs_inj_GCE, 0.84, axis=1) - np.quantile(FFs_inj_GCE, 0.5, axis=1)
    yerr = np.vstack([yerr_low, yerr_high])
    ax.errorbar(100 * np.asarray(xi_vec), y=100 * FF_median, ls="none", yerr=100 * yerr, capsize=2, ecolor="k",
                 marker=".", ms=0, markeredgewidth=1, elinewidth=2, mec="deepskyblue", mfc="white", barsabove=True)

    # Plot injected FF that corresponds to upper prior limit
    upper_prior_limit_DM_flux = 2.8e-07
    upper_prior_limit_DM_counts = upper_prior_limit_DM_flux * generator_test.settings_dict["exp"].mean()
    Delta_f_lim = upper_prior_limit_DM_flux - GCE_flux_orig
    Delta_xi_lim = Delta_f_lim / (Delta_f_lim + total_fermi_flux)
    # Check correctness
    print("Delta_xi_lim:", Delta_xi_lim)
    print("This gives:", (Delta_xi_lim / (1 - Delta_xi_lim) * total_fermi_flux + GCE_flux_orig) * generator_test.settings_dict["exp"].mean(),
          "GCE counts for the injection map at Delta xi lim.")
    rect = mpl.patches.Rectangle([100 * Delta_xi_lim, -0.5 + 100 * FFs[GCE_ind]], 100 * exp_FF_xi_max + 0.5 - Delta_xi_lim,
                                 100 * exp_FF_xi_max - 100 * FFs[GCE_ind] + 1, fill=True, color="k", ec="none", alpha=0.2)
    ax.add_patch(rect)
    # ax.axvline(100 * Delta_xi_lim, color="k", alpha=0.3, linestyle=":")
    ax.set_xlim([-0.5, 100 * max_xi + 0.5])
    ax.set_ylim([-0.5 + 100 * FFs[GCE_ind], 100 * exp_FF_xi_max + 0.5])
    ax.set_aspect("equal")
    ax.set_xlabel(r"Injected")
    ax.set_ylabel(r"Estimated")
    plt.tight_layout()
    fig_name = "Fermi_DM_injection_FF_plot.pdf"
    fig_ff.savefig(os.path.join(model.get_path("checkpoints"), fig_name), bbox_inches="tight")

    # 2) Plot for the predicted histograms
    # inj_hist_filename = os.path.join(model.get_path("checkpoints"), "injection_plot_hists.pdf")
    inj_hist_filename = ""
    mean_exp = generator_test.settings_dict["exp"].mean()
    xi_inds_to_plot = [1, 3, 5]
    xis_to_plot = xi_vec[xi_inds_to_plot]
    fermi_pred_tiled = np.tile(np.expand_dims(fermi_pred["gce_hist"], 1), [1, n_maps, 1, 1])[None]   # first: plot Fermi prediction without injection (need to tile)
    injection_input = np.concatenate([fermi_pred_tiled, all_hists_inj[xi_inds_to_plot, :, :, :, :]], axis=0)
    tau_inds_ebars = [0, 9, 18]  # indices for which scatter over the samples is shown in cum. histogram
    make_plot_injection(injection_input, tau_vec, tau_inds_ebars, bin_centres, save_name=inj_hist_filename, mean_exp=mean_exp)

    # 3) Plot the median CDFs in a single axis
    fig_cdfs, ax_cdfs = plt.subplots(1, 1, figsize=(4.48, 4.09))
    width = np.diff(bin_centres)[0]
    median_ind = all_hists_inj.shape[1] // 2
    colors = cc.cm.bgy(np.linspace(0, 1, n_xi + 1))

    # Now: injection
    for i_xi in range(len(xi_vec) - 1, -1, -1):
        ax_cdfs.step(bin_centres - width / 2, np.median(all_hists_inj[i_xi, median_ind, :, :, 0].cumsum(-1), 0),  where="post",
                     color=colors[i_xi + 1])

    # Plot CDF for original Fermi map
    ax_cdfs.step(bin_centres - width / 2, fermi_pred["gce_hist"][median_ind, :, 0].cumsum(), where="post", color="k")

    # Plot settings
    ax_cdfs.set_xlim([-13, -9])
    one_ph_flux = np.log10(1 / mean_exp)
    ax_cdfs.axvline(one_ph_flux, color="orange", ls="--", zorder=4)
    rect = mpl.patches.Rectangle((np.log10(4e-10), -0.075), np.log10(5e-10) - np.log10(4e-10), 1.075 + 0.075,
                                 linewidth=0, edgecolor=None, facecolor="#cccccc", zorder=-1)
    ax_cdfs.add_patch(rect)

    # Build twin axes and set limits
    def F2S(x):
        return 10.0 ** x * mean_exp
    twin_axes = ax_cdfs.twiny()
    twin_axes.plot(F2S(bin_centres), fermi_pred["gce_hist"][median_ind, :, 0].cumsum(), color="none", lw=0)
    twin_axes.set_xlim(F2S(np.asarray([-13, -9])))
    twin_axes.set_xlabel(r"$\bar{S}$")
    twin_axes.set_xscale("log")
    twin_axes.set_ylim([-0.075, 1.075])
    ax_cdfs.set_xlabel(r"$\log_{10} \ F$")
    ax_cdfs.set_ylabel("CDF")
    plt.tight_layout()
    fig_name = "Fermi_DM_injection_CDF_median_plot.pdf"
    fig_cdfs.savefig(os.path.join(model.get_path("checkpoints"), fig_name), bbox_inches="tight")
