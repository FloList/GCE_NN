"""
This script evaluates the NN for the isotropic example.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from make_tile_plot import make_tile_plot
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
    TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    bin_edges = copy.copy(params["gce_hist_bins"])

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Get test data
    n_samples_analysis = 256
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
    band_mask_range = 0

    # 1) Make histogram plot of test data
    n_plot = 9
    # Get n_plot maps that cover a large range of histograms
    sort_val = 0.95
    inds_sorted = np.argsort(np.argmin((test_out["gce_hist"][:, :, 0].cumsum(1) < sort_val), axis=1))
    plot_samples = inds_sorted[np.floor(np.linspace(0, n_samples_analysis - 1, n_plot)).astype(int)]
    # re-sort: row/col -> col/row
    plot_samples = np.reshape(plot_samples, [3, 3]).T.flatten()
    make_tile_plot(model, tau_vec, bin_centres, params, test_out, plot_samples=plot_samples, inner_band=band_mask_range,
                        name="tile_plot.pdf", name_maps="tile_plot_maps.pdf", mean_exp=mean_exp)
