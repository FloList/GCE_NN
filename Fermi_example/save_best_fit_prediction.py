"""
This script saves the NN prediction for simulated Fermi best-fit maps.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
import os
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

    # Define quantile levels tau
    tau_vec = np.linspace(0.05, 0.95, 19)
    n_taus = len(tau_vec)

    # Load best-fit data
    out_files = ["best_fit_pred", "best_fit_smooth_GCE_pred"]
    for i_file, file in enumerate(["best_fit_maps", "best_fit_maps_smooth_GCE"]):
        try:
            best_fit_data = np.load(os.path.join(model.get_path("checkpoints"), file + ".npy"), allow_pickle=True)[()]
        except FileNotFoundError:
            raise FileNotFoundError("Run the script 'generate_Fermi_best_fit_maps.py' first!")

        best_fit_maps = best_fit_data["data"]
        n_maps = best_fit_data["data"].shape[0]

        best_fit_pred_all = dict()

        # Iterate over quantile levels
        for i_tau, tau in enumerate(tau_vec):
            test_dict = {"data": best_fit_maps}

            # Predict
            best_fit_pred = model.predict(test_dict, None, tau_hist=tau * np.ones((n_maps, 1)))

            # Store
            for key in best_fit_pred.keys():
                if key not in best_fit_pred_all:
                    best_fit_pred_all[key] = best_fit_pred[key][None]
                else:
                    best_fit_pred_all[key] = np.concatenate([best_fit_pred_all[key], best_fit_pred[key][None]], axis=0)

        # Delete map data (would result in huge files)
        for key in ["count_maps", "count_maps_modelled_Poiss", "count_maps_residual", "input_hist_channel_2"]:
            del best_fit_pred_all[key]

        # Save
        np.save(os.path.join(model.get_path("checkpoints"), out_files[i_file]), best_fit_pred_all)
        print("SAVED: ", os.path.join(model.get_path("checkpoints"), out_files[i_file]))

    # # Loading
    # best_fit_pred = np.load(os.path.join(model.get_path("checkpoints"), out_files[0] + ".npy"), allow_pickle=True)[()]
    # best_fit_pred_smooth = np.load(os.path.join(model.get_path("checkpoints"), out_files[1]  + ".npy"), allow_pickle=True)[()]
