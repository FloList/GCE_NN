"""
Analyse the NN predictions for simulated Fermi best-fit maps (need to run 'save_best_fit_prediction' first).
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

    # Load Fermi prediction
    try:
        fermi_pred_data = np.load(os.path.join(model.get_path("checkpoints"), "fermi_prediction.npz"),
                                  allow_pickle=True)
        print(fermi_pred_data.files)
        fermi_pred = fermi_pred_data["fermi_pred"][()]
        bin_centres = fermi_pred_data["bin_centres"][()]
        assert np.all(fermi_pred_data["tau_vec"][()] == tau_vec)
    except FileNotFoundError:
        raise FileNotFoundError("Run the script 'save_fermi_prediction' first!")

    # Load best-fit predictions
    pred_files = ["best_fit_pred", "best_fit_smooth_GCE_pred"]
    try:
        best_fit_pred = np.load(os.path.join(model.get_path("checkpoints"), pred_files[0] + ".npy"), allow_pickle=True)[()]
        best_fit_pred_smooth = np.load(os.path.join(model.get_path("checkpoints"), pred_files[1] + ".npy"), allow_pickle=True)[()]
    except FileNotFoundError:
        raise FileNotFoundError("Run the script 'save_best_fit_prediction.py' first!")

    # First: FFs
    print("Flux fractions (median for best-fit maps):")
    FFs_fermi = fermi_pred["logits_mean"][0]
    FFs_best_fit_median = np.median(best_fit_pred["logits_mean"], 1).mean(0)
    FFs_best_fit_median_smooth = np.median(best_fit_pred_smooth["logits_mean"], 1).mean(0)
    print("  Fermi:              ", FFs_fermi)
    print("  Best fit:           ", FFs_best_fit_median)
    print("  Best fit smooth GCE:", FFs_best_fit_median_smooth)

    # Now: GCE histograms
    hist_fermi = fermi_pred["gce_hist"][:, :, 0]
    hist_best_fit = best_fit_pred["gce_hist"]

    # Plot settings
    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # Make a plot of the CDFs and of the FF constraints
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    ax = axs[0]
    # Plot the Fermi prediction: median and 5-95% shaded
    median_ind = (n_taus - 1) // 2
    ax.plot(bin_centres, fermi_pred["gce_hist"][median_ind, :, 0].cumsum(), "k-", lw=2)
    ax.fill_between(bin_centres, fermi_pred["gce_hist"][-1, :, 0].cumsum(), fermi_pred["gce_hist"][0, :, 0].cumsum(),
                        color="k", alpha=0.15)

    # Plot the Fermi prediction for best-fit maps (median over the maps)
    smooth_median_hist_median_cum = np.quantile(best_fit_pred_smooth["gce_hist"][median_ind, :, :, 0].cumsum(1), 0.5, 0)
    smooth_median_hist_5_cum = np.quantile(best_fit_pred_smooth["gce_hist"][median_ind, :, :, 0].cumsum(1), 0.05, 0)
    smooth_median_hist_95_cum = np.quantile(best_fit_pred_smooth["gce_hist"][median_ind, :, :, 0].cumsum(1), 0.95, 0)
    ax.plot(bin_centres, smooth_median_hist_median_cum, "-", lw=2, color="firebrick")
    ax.fill_between(bin_centres, smooth_median_hist_95_cum, smooth_median_hist_5_cum, color="firebrick", alpha=0.15)
