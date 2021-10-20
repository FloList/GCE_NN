"""
Get number of PSs from a SCD
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
from scipy.integrate import simps, trapz
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


def process_templates(template_path, models, indices, exp_name="fermidata_exposure", hist_templates=None):
    """
    :param template_path: path to the template folder
    :param models: list of template names
    :param indices: indices to use from templates: FROM RING TO NESTED!
    :param exp_name: name of exposure template if exposure correction shall be removed
    :param hist_templates: list of templates names for which histogram will be computed.
                           They will not be modelled away to obtain residual.
    """
    template_dict = dict()

    # Get names of the templates to load (might need to remove trailing _PS)
    model_names_without_PS = [model if "PS" not in model else model[:-3] for model in models]

    if hist_templates is None:
        template_dict["temp_indices"] = np.arange(len(models))
    else:
        template_dict["temp_indices"] = np.argwhere([model not in hist_templates for model in models]).flatten()

    # Get exposure and convert to NEST format
    fermi_exp_full = hp.reorder(np.load(os.path.join(template_path, exp_name + ".npy")), r2n=True)
    fermi_exp_full_mean = fermi_exp_full.mean()
    # Calculate rescale: NOTE: calculated on UNMASKED ROI!!
    fermi_rescale = (fermi_exp_full / fermi_exp_full_mean)[indices]

    # Get relevant indices
    template_dict["fermi_exp"] = fermi_exp_full[indices]
    template_dict["fermi_rescale"] = fermi_rescale

    template_dict["exp"] = template_dict["fermi_exp"]
    template_dict["rescale"] = template_dict["fermi_rescale"]

    n_pix_ROI = template_dict["fermi_exp"].shape[0]
    n_models = len(models)
    template_dict["T_counts"] = np.zeros((n_models, n_pix_ROI))
    template_dict["T_flux"] = np.zeros((n_models, n_pix_ROI))
    template_dict["counts_to_flux_ratio"] = np.zeros((n_models))

    # Iterate over the templates
    for i_name, name in enumerate(model_names_without_PS):
        temp_counts = hp.reorder(get_template(template_path, name), r2n=True)
        temp_counts = temp_counts[indices]
        # remove exposure correction to go to "flux space"
        temp_flux = temp_counts / fermi_rescale
        counts_to_flux_ratio = temp_counts.sum() / temp_flux.sum()
        template_dict["T_counts"][i_name, :] = temp_counts
        template_dict["T_flux"][i_name, :] = temp_flux
        template_dict["counts_to_flux_ratio"][i_name] = counts_to_flux_ratio

    return template_dict


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
    nside = params["nside"]

    if GADI:
        sys.path.append('/scratch/u95/fl9575/GCE_v2/Fermi_example/nside_' + str(nside))
    from ps_mc_fast_entire_sky_restricted import run

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

    # Get FFs, fluxes, etc.
    total_fermi_counts = int((generator_test.settings_dict["rescale"] * fermi_counts).sum())  # total counts in ROI
    exp_compressed_nest = generator_test.settings_dict["exp"][generator_test.settings_dict["unmasked_pix"]]
    total_fermi_flux = (generator_test.settings_dict["rescale"] * fermi_counts / exp_compressed_nest).sum()
    FFs = fermi_pred["logits_mean"].mean(0)  # avg. over taus (identical)
    fluxes = FFs * total_fermi_flux

    # Build a dictionary with template information
    template_dict = process_templates(params["template_path"], params["models"], params["indexes"][0],
                                      hist_templates=[4, 5])

    # Flux fractions -> count fractions
    FF_sum = FFs.sum()
    count_fracs_unnorm = FFs * template_dict["counts_to_flux_ratio"]
    count_fracs = count_fracs_unnorm / (np.sum(count_fracs_unnorm, keepdims=True) / FF_sum)
    print("FFs:", FFs, "\nCFs:", count_fracs)

    # Get counts per template
    # fermi_count_map = fermi_counts * template_dict["rescale"]
    # assert int(fermi_count_map.sum()) == total_fermi_counts, "Total number of counts in Fermi map is incorrect!"
    counts_per_temp = total_fermi_counts * count_fracs

    # Get ratio between counts per template and template sum
    T_counts_rescale_fac = counts_per_temp / template_dict["T_counts"].sum(1)

    # Initialise best-fit maps
    n_maps = 256
    n_pix_ROI = len(params["indexes"][0])
    best_fit_maps_templates = np.zeros((len(params["models"]), n_maps, n_pix_ROI))

    # Exposure map (uncompressed)
    fermi_exp = get_template(fermi_folder, "exp")
    fermi_exp = hp.reorder(fermi_exp, r2n=True)
    mean_exp = np.mean(fermi_exp)
    fermi_rescale = fermi_exp / mean_exp

    # Quadrature rule
    int_fun = trapz  # trapz, simps, ...

    temp = "gce_12_PS"
    i_temp = 4  # index of GCE template
    temp_ind_hist = 0 if "gce" in temp else 1
    T = get_template(fermi_folder, temp[:-3])  # here, template should NOT be compressed -> do not use template_dict
    T = hp.reorder(T, r2n=True)
    T_corr = T / fermi_rescale

    # Do NOT apply mask, rather rescale!
    total_mask_neg = np.ones(hp.nside2npix(params["nside"]))
    total_mask_neg[params["indexes"][0]] = 0

    # Template needs to be normalised to sum up to unity for the new implementation!
    # Might need to do this twice because of rounding errors
    T_final = T_corr / T_corr.sum()
    while T_final.sum() > 1.0:
        T_final /= T_final.sum()
    if T_final.sum() != 1.0:
        print("WARNING: TEMPLATE SUM IS NOT EXACTLY 1 BUT", T_final.sum(), "!")

    T_final_masked = T_final * (1 - total_mask_neg)
    area_template_frac = 1.0 / T_final_masked.sum()

    # only consider brightest sources starting from a certain index?
    ind_min_F = 0
    if ind_min_F > 0:
        print("CUTTING OF SCD BELOW ind_min_F =", ind_min_F, "AND RESCALING!")

    # Iterate over taus
    print("Expected number of PSs:")
    for i_tau, tau in enumerate(tau_vec):
        print("  tau: {:1.2f}".format(tau))
        # relative F dN/dF -> dN/dF with correct scaling
        F_ary = 10 ** bin_centres
        F_tot_template = FFs[i_temp] * total_fermi_flux
        F_tot_template_entire_map = F_tot_template * area_template_frac
        if i_tau == 0:
            print("Total flux from template (ROI):", F_tot_template)
            print("Total flux from template (sky):", F_tot_template_entire_map)

        hist_rel = fermi_pred["gce_hist"][i_tau, :, temp_ind_hist].copy()

        if ind_min_F > 0:
            hist_rel[:ind_min_F] = 0
            hist_rel /= hist_rel.sum()

        # The histogram is normalised such that sum(histogram_rel) = 1
        # sum(histogram_abs) = F_tot, where histogram_rel = histogram_abs / F_tot
        hist_abs_all_sky = hist_rel * F_tot_template_entire_map

        # Scale such that integral over log10(F) gives F_tot, rather than the sum
        int_hist_abs_all_sky = int_fun(hist_abs_all_sky, bin_centres)
        FdNdlogF_all_sky = hist_abs_all_sky / int_hist_abs_all_sky * F_tot_template_entire_map
        # now, int_fun(FdNdlogF, bin_centres) = F_tot_template_entire_map

        # From FdNdlogF, compute number of expected PSs via
        #  N_exp = int dN/dlogF dlogF
        N_tot_template_all_sky = int_fun(FdNdlogF_all_sky / F_ary, bin_centres)
        N_tot_template_ROI = N_tot_template_all_sky / area_template_frac
        print("    entire sky: {:7.0f}".format(N_tot_template_all_sky))
        print("    ROI:        {:7.0f}".format(N_tot_template_ROI))

        # How many PS are needed to explain a fraction of the flux?
        for i in range(len(bin_centres) - 1, -1, -1):
            F_tot_template_all_sky_cum_high2low = int_fun(FdNdlogF_all_sky[i:], bin_centres[i:])
            N_tot_template_all_sky_cum_high2low = int_fun((FdNdlogF_all_sky / F_ary)[i:], bin_centres[i:])
            N_tot_template_ROI_cum_high2low = N_tot_template_all_sky_cum_high2low / area_template_frac
            min_flux_threshold = bin_edges[i]
            min_counts_threshold = min_flux_threshold * mean_exp
            print("         {:.0f}".format(N_tot_template_all_sky_cum_high2low),
                  "PSs in the entire sky / {:.0f} in the ROI".format(N_tot_template_ROI_cum_high2low),
                  "explain {:3.1f}%".format(100 * F_tot_template_all_sky_cum_high2low / F_tot_template_entire_map),
                  "of the flux (min. exp. counts: {:3.3f} / flux: {:3.3g}).".format(min_counts_threshold, min_flux_threshold))
