"""
    Evaluate trained NN on mismodelled maps.
    # Cases of mismodelling considered in this script:
    # 0: original, no mismodelling
    # 1: Thin disk -> thick disk
    # 2: Fermi bubbles -> Fermi bubbles variant
    # 3: Diffuse Model O -> Model A
    # 4: Diffuse Model O -> Model F
    # 5: Diffuse Model O -> p6v11
    # 6: GCE with gamma = 1.2 -> GCE with gamma = 1.0
    # 7: Asymmetric GCE (N: 2/3, S: 1/3)
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from make_plot_mismodelling import make_plot_mismodelling
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
    data_out_file = "evaluate_NN_mismodelling_data"

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Load mismodelling data
    mismodelling_file = os.path.join(model.get_path("checkpoints"), "best_fit_maps_mismodelling.npy")
    if not os.path.exists(mismodelling_file):
        raise FileNotFoundError("Run 'generate_Fermi_best_fit_maps_mismodelling.py' first!")
    mismodelling_dict = np.load(mismodelling_file, allow_pickle=True)
    data_in = mismodelling_dict[()]["data"]
    label = mismodelling_dict[()]["label"]
    hists = mismodelling_dict[()]["gce_hist"]

    data_flat = np.reshape(data_in, [np.product(data_in.shape[:2])] + [-1])  # flatten mismodelling cases x realisations
    n_samples = data_flat.shape[0]
    samples = np.arange(n_samples)

    # Define quantile levels tau
    tau_vec = np.linspace(0.05, 0.95, 19)
    n_taus = len(tau_vec)

    # Get mean exposure
    mean_exp = generator_test.settings_dict["exp"].mean()

    # Predict
    if not os.path.exists(os.path.join(model.get_path("checkpoints"), data_out_file + ".npy")):
        dict_all = dict()
        for i_tau, tau in enumerate(tau_vec):
            NN_pred = model.predict({"data": data_flat[:n_samples, :]}, None, False, tau_hist=tau * np.ones((n_samples, 1)))
            # NN_pred = dict()
            # NN_pred["gce_hist"] = np.ones((n_samples, len(bin_centres), 2))
            # NN_pred["logits_mean"] = np.ones((n_samples, 6))
            # NN_pred["covar"] = np.ones((n_samples, 6, 6))

            for key in NN_pred.keys():
                if key not in dict_all.keys():
                    dict_all[key] = []
                dict_all[key].append(NN_pred[key])

        for key in dict_all.keys():
            dict_all[key] = np.asarray(dict_all[key])
            # Transpose: move quantile level dimension to the end before un-flattening
            dict_all[key] = dict_all[key].transpose(*(list(np.arange(1, len(dict_all[key].shape))) + [0]))
            # Unflatten the mismodelling cases
            dict_all[key] = np.reshape(dict_all[key], list(data_in.shape[:2]) + list(dict_all[key].shape[1:]))
            # Output shape: n_mismodelling_cases x n_taus x n_realisations x *(property_shape_for_single_map)
            dict_all[key] = dict_all[key].transpose(*([0, len(dict_all[key].shape) - 1] + list(np.arange(1, len(dict_all[key].shape) - 1))))

        for key in ['count_maps', 'count_maps_modelled_Poiss', 'count_maps_residual', 'input_hist_channel_2']:
            del dict_all[key]

        np.save(os.path.join(model.get_path("checkpoints"), data_out_file), dict_all)
        print("Computation done - predictions saved!")
        sys.exit(0)

    else:
        data = np.load(os.path.join(model.get_path("checkpoints"), data_out_file + ".npy"), allow_pickle=True)[()]
        pred_hist_all = data["gce_hist"]  # n_mismodelling_cases x n_taus x n_realisations x n_bins x 2 (GCE, disk)
        pred_FF_all = data["logits_mean"]  # n_mismodelling_cases x n_taus x n_realisations x n_templates
        pred_covar_all = data["covar"]  # n_mismodelling_cases x n_taus x n_realisations x n_templates x n_templates

    # Get the cumulative histograms
    pred_hist_all_cum = pred_hist_all.cumsum(3)

    # Get the median over the maps, for each mismodelling case, quantile level, and disk & GCE
    pred_hist_all_cum_median = np.median(pred_hist_all_cum, 2)
    pred_hist_all_cum_median_delta = np.diff(pred_hist_all_cum_median, axis=2)
    first_val_median = 1 - pred_hist_all_cum_median_delta.sum(2, keepdims=True)  # this is pred_hist_all_cum_median in the first bin
    pred_hist_all_median = np.concatenate([first_val_median, pred_hist_all_cum_median_delta], axis=2)

    pred_FF_median = np.median(pred_FF_all.mean(1), 1)  # FFs are independent of tau
    pred_covar_median = np.median(pred_covar_all.mean(1), 1)  # FF covariances are independent of tau
    pred_std_median = np.asarray([np.sqrt(np.diag(cov)) for cov in pred_covar_median])

    # Get truth
    # EXCLUDE p6v11 HERE WHEN COMPUTING THE MEDIAN FFS AS IT HAS ALL THE DIFFUSE FLUX STORED IN DIF PIBS
    true_FFs = np.median(label[[0, 1, 2, 3, 4, 6, 7], :, :], 1).mean(0)
    print("Sum of avg. true FFs:", np.median(label[[0, 1, 2, 3, 4, 6, 7], :, :], 1).mean(0).sum())

    true_hists = np.median(hists, 1).mean(0)
    print("Sum of avg. true hists:", np.median(hists, 1).mean(0).sum(0))

    for is_cdf in [True, False]:
        if is_cdf:
            name = "mismodelling_plot_cdf_w_truth.pdf"
        else:
            name = "mismodelling_plot_pdf_w_truth.pdf"
        filename = os.path.join(model.get_path("checkpoints"), name)
        # filename = ""
        make_plot_mismodelling(pred_FF_median, pred_std_median, pred_hist_all_median, tau_vec, bin_centres, params,
                               filename=filename, mean_exp=mean_exp, cum=is_cdf, true_FFs=true_FFs, true_hists=true_hists,
                               colours=None, exclude_FF_inds=[5])
