"""
Evaluate NN on maps I_1 and I_2 (reshuffled version of I_1) and make a plot.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
import os
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper")
plt.ion()
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
    TEST_EXP_PATH = "./checkpoints/NS_maps_combined_l2_IN_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    save_folder = '/scratch/u95/fl9575/GCE_v2/North_south_mismodelling/Data'
    save_filename = os.path.join(save_folder, "data")
    dl = np.load(save_filename + ".npz")
    I_1, I_2, sigma, compression_inds = dl["I_1"], dl["I_2"], dl["sigma"], dl["compression_inds"]
    I_1_comp, I_2_comp = I_1[compression_inds], I_2[compression_inds]
    I_1_nest, I_2_nest = hp.reorder(I_1, r2n=True), hp.reorder(I_2, r2n=True)
    compression_inds_nest = generator_test.settings_dict["unmasked_pix"]
    I_1_nest_comp, I_2_nest_comp = I_1_nest[compression_inds_nest], I_2_nest[compression_inds_nest]
    assert I_1_nest_comp.sum() == I_1.sum(), I_2_nest_comp.sum() == I_2.sum()

    I_1_pred = model.predict({"data": I_1_nest_comp[None]})
    I_2_pred = model.predict({"data": I_2_nest_comp[None]})
    print(I_1_pred, I_2_pred)

    # Tests with purely Poissonian maps
    # lambdas = [1, 2, 5, 10]
    # I_poiss = np.asarray([np.random.poisson(lam, size=I_1_nest_comp.shape) for lam in lambdas])
    # I_poiss_pred = model.predict({"data": I_poiss})

    # save_pred_filename = os.path.join(save_folder, "NN_pred")
    # np.savez(save_pred_filename, I_1_pred=I_1_pred["logits_mean"], I_2_pred=I_2_pred["logits_mean"])

    # Plot flux fractions
    colour_P = 'deepskyblue'
    colour_NP = 'darkslateblue'
    for pred, tag in zip([I_1_pred, I_2_pred], ['NS_asymmetry', 'NS_shuffled']):
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(3, 3))
        ax_1.axvline(100 * pred["logits_mean"][0, 0], color=colour_P, label="P", lw=3)
        ax_1.axvline(100 * pred["logits_mean"][0, 1], color=colour_NP, label="PS", lw=3)
        ax_1.set_xlabel('Flux fraction (%)')
        ax_1.legend(fancybox=True)
        ax_1.set_xlim(0, 100)
        ax_1.set_ylim(0, .1)
        plt.tight_layout()
        fig_1.savefig("FF_NN_" + tag + ".pdf")
