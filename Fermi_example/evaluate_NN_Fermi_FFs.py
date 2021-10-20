"""
Evaluate trained NN for the Fermi example on simulated maps (only FFs).
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
from make_error_plot_zoomed import make_error_plot_zoomed
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

    # Get test data
    n_samples_analysis = 256
    models_test = params["models"]
    # model_names = params["model_names"]
    model_names = ['Diffuse $\\pi^0$ + BS', 'Diffuse IC', 'Isotropic', '$\\it{Fermi}$ bubbles', 'GCE', 'Disk']
    sess_test = None if TEST_CHECKPOINT is None else model.get_session(checkpoint=TEST_CHECKPOINT)
    input_test_dict = ds_test.next_element.vars()
    test_out = ds_test.get_samples(n_samples_analysis)
    test_data, test_label = test_out["data"], test_out["label"]
    if "gce_hist" in test_out.keys():
        test_gce_hist = test_out["gce_hist"]
    real_fluxes = test_label

    # Get mean exposure and set band mask range
    mean_exp = generator_test.settings_dict["exp"].mean()
    band_mask_range = 2

    # Make flux fraction plot
    NN_pred = model.predict(test_out)
    ms = None
    covar_fluxes = NN_pred["covar"]
    colours = ['#ff0000', '#ec7014', '#fec400', '#37c837', 'deepskyblue', 'k']
    ax_lims = [[0.285, 0.715], [0.185, 0.485], [0.0, 0.135], [0.0, 0.135], [0.0, 0.185], [0.0, 0.185]]
    ticks = [[0.3, 0.4, 0.5, 0.6, 0.7], [0.2, 0.3, 0.4], [0.0, 0.05, 0.1], [0.0, 0.05, 0.1],
             [0.0, 0.05, 0.1, 0.15], [0.0, 0.05, 0.1, 0.15]]
    tick_labels = [[str(np.round(100 * t).astype(int)) for t in tick] for tick in ticks]
    make_error_plot_zoomed(models_test, ax_lims, ticks, tick_labels, real_fluxes, NN_pred["logits_mean"],
                           model_names=model_names, out_file="flux_fraction_scatter_plot.pdf", show_stats=True,
                           colours=colours, pred_covar=covar_fluxes, ms=ms, show_ticks=True, marker=".")
