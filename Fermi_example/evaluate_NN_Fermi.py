"""
Evaluate trained NN for the Fermi example on simulated maps (only SCDs) and on the Fermi data.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from make_plot_simulated_tiles import make_plot_simulated_tiles
from make_plot_fermi import make_plot_fermi
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
    data_out_file = "evaluate_NN_Fermi_data"

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Get test data
    n_samples_analysis = 64
    models_test = params["models"]
    model_names = params["model_names"]
    sess_test = None if TEST_CHECKPOINT is None else model.get_session(checkpoint=TEST_CHECKPOINT)
    input_test_dict = ds_test.next_element.vars()
    test_out = ds_test.get_samples(n_samples_analysis)
    test_data, test_label = test_out["data"], test_out["label"]
    if "gce_hist" in test_out.keys():
        test_gce_hist = test_out["gce_hist"]
    real_fluxes = test_label

    # Define quantile levels tau
    tau_vec = np.linspace(0.05, 0.95, 19)
    n_taus = len(tau_vec)

    # Get mean exposure and set band mask range
    mean_exp = generator_test.settings_dict["exp"].mean()
    band_mask_range = 2

    # 1) Make histogram plot of test data
    # make_plot_simulated(model, tau_vec, bin_centres, params, test_out, plot_samples=plot_inds, inner_band=band_mask_range,
    #                     name="simulated_plot.pdf", mean_exp=mean_exp, residual_clim_fac=4)

    # Get n_plot maps that cover a large range of histograms
    n_plot = 8
    sort_val = 0.95
    sort_ch = 0  # sort by GCE brightness
    inds_sorted = np.argsort(np.argmin((test_out["gce_hist"][:, :, sort_ch].cumsum(1) < sort_val), axis=1))
    plot_samples = inds_sorted[np.floor(np.linspace(0, n_samples_analysis - 1, n_plot)).astype(int)]

    # Predict
    if not os.path.exists(os.path.join(model.get_path("checkpoints"), data_out_file + ".npy")):
        dict_all = dict()

        for i_sample, db_sample in enumerate(plot_samples):
            # Evaluate NN
            test_out_loc = copy.copy(test_out)

            # Tile for the different quantile levels
            for key in test_out.keys():
                test_out_loc[key] = np.tile(test_out[key][db_sample, :][None], [n_taus] + [1] * (len(test_out[key].shape) - 1))

            # Predict and get means and histograms
            pred_fluxes_dict = model.predict(test_out_loc, None, tau_hist=np.expand_dims(tau_vec, -1))
            for key in pred_fluxes_dict.keys():
                if key not in dict_all.keys():
                    dict_all[key] = []
                dict_all[key].append(pred_fluxes_dict[key])

        for key in dict_all.keys():
            dict_all[key] = np.asarray(dict_all[key])

        np.save(os.path.join(model.get_path("checkpoints"), data_out_file), dict_all)
        print("Computation done - predictions saved!")

    else:
        data = np.load(os.path.join(model.get_path("checkpoints"), data_out_file + ".npy"), allow_pickle=True)[()]
        pred_hist_all = data["gce_hist"]
        pred_FF_all = data["logits_mean"]
        pred_covar_all = data["covar"]

        # # Plot maps
        # palette = copy.copy(sns.cm.rocket_r)
        # palette.set_bad(color='white', alpha=0)
        #
        # total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=band_mask_range, mask_ring=True, inner=0,
        #                                     outer=params["outer_rad"], nside=params["nside"])
        # if params["mask_type_fermi"] == "3FGL":
        #     total_mask_neg = (
        #                 1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "3FGL_mask"))).astype(
        #         bool)
        # elif params["mask_type_fermi"] == "4FGL":
        #     total_mask_neg = (
        #                 1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "4FGL_mask"))).astype(
        #         bool)
        # total_mask_neg = hp.reorder(total_mask_neg, r2n=True)
        # max_val = data["count_maps"].max() / 20
        #
        # for i_sample, db_sample in enumerate(plot_samples[:2]):
        #     count_map = data["count_maps"][i_sample, :].mean(0)[:, 0]  # avg. over taus (identical)
        #     plot_data_full = masked_to_full(count_map, params["indexes"][0], nside=params["nside"])
        #     plot_data_full[total_mask_neg] = np.nan
        #     hp.cartview(plot_data_full, nest=True, title=str(i_sample), cbar=False, max=max_val, min=0,
        #                 cmap=palette)

    # 1) Make plot of simulated maps
    make_plot_simulated_tiles(model, tau_vec, bin_centres, params, test_out, plot_samples=plot_samples, layout=(2, 4),
                              name="simulated_plot_tiles.pdf", mean_exp=mean_exp, hist_inds=[4, 5],
                              pred_hist_all=pred_hist_all)

    # 2) Fermi prediction
    make_plot_fermi(model, tau_vec, bin_centres, params, {"data": fermi_counts[None]}, inner_band=band_mask_range,
                       name="fermi_plot.pdf", mean_exp=mean_exp, residual_clim_fac=4)
