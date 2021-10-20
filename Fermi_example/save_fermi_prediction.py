"""
This script saves the NN prediction for the Fermi data.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
from gce_utils import *
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

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Define quantile levels tau
    tau_vec = np.linspace(0.05, 0.95, 19)
    n_taus = len(tau_vec)

    # Tile Fermi data for different quantile levels
    test_out = {"data": fermi_counts[None]}
    test_out_tiled = dict()
    for key in test_out.keys():
        test_out_tiled[key] = np.tile(test_out[key][0, :][None], [n_taus] + [1] * (len(test_out[key].shape) - 1))

    # Predict
    fermi_pred = model.predict(test_out_tiled, None, tau_hist=np.expand_dims(tau_vec, -1))

    # Save
    np.savez(os.path.join(model.get_path("checkpoints"), "fermi_prediction"), fermi_pred=fermi_pred, tau_vec=tau_vec,
             bin_centres=bin_centres)

    # Loading
    fermi_pred_data = np.load(os.path.join(model.get_path("checkpoints"), "fermi_prediction.npz"), allow_pickle=True)
    print(fermi_pred_data.files)
    fermi_pred = fermi_pred_data["fermi_pred"][()]
    bin_centres = fermi_pred_data["bin_centres"][()]
    tau_vec = fermi_pred_data["tau_vec"][()]
