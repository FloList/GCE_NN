"""
Generate simulated maps corresponding to the Fermi best-fit parameters as determined by the NN.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
import scipy as sp
from scipy.integrate import simps, trapz
import os
import copy
import seaborn as sns
import tqdm
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
    template_dict["counts_to_flux_ratio"] = np.zeros(n_models)

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


# PDF Sampler
class PDFSampler:
    def __init__(self, xvals, pofx):
        """ At outset sort and calculate CDF so not redone at each call

            :param xvals: array of x values
            :param pofx: array of associated p(x) values (does not need to be
                   normalised)
        """
        self.xvals = xvals
        self.pofx = pofx

        # Check p(x) >= 0 for all x, otherwise stop
        assert(np.all(pofx >= 0)), "pdf cannot be negative"

        # Sort values by their p(x) value, for more accurate sampling
        self.sortxvals = np.argsort(self.pofx)
        self.pofx = self.pofx[self.sortxvals]

        # Calculate cdf
        self.cdf = np.cumsum(self.pofx)

    def __call__(self, samples):
        """ When class called returns samples number of draws from pdf

            :param samples: number of draws you want from the pdf
            :returns: number of random draws from the provided PDF

        """

        # Random draw from a uniform, up to max of the cdf, which need
        # not be 1 as the pdf does not have to be normalised
        unidraw = np.random.uniform(high=self.cdf[-1], size=samples)
        cdfdraw = np.searchsorted(self.cdf, unidraw)
        cdfdraw = self.sortxvals[cdfdraw]
        return self.xvals[cdfdraw]


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
    data_out_file = "best_fit_maps"
    nside = params["nside"]
    SMOOTH_GCE = False  # if True: replace PS GCE by a purely Poissonian GCE of the same intensity

    if SMOOTH_GCE:
        data_out_file += "_smooth_GCE"

    if SMOOTH_GCE:
        print("SMOOTH GCE!")
    else:
        print("PS GCE!")

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
    n_maps = 1024
    n_pix_ROI = len(params["indexes"][0])
    best_fit_maps_templates = np.zeros((len(params["models"]), n_maps, n_pix_ROI))

    # Start with the Poissonian templates
    if SMOOTH_GCE:
        T_P, T_NP = params["models"][:4] + ["gce_12"], params["models"][-1:]  # LEAVE OUT gce_12_PS and replace by gce_12
    else:
        T_P, T_NP = params["models"][:4], params["models"][4:]

    for i_temp, temp in enumerate(T_P):
        print("Starting with Poissonian model", temp)
        if i_temp < 4:
            T = template_dict["T_counts"][i_temp]  # get counts template
        else:
            assert SMOOTH_GCE, "Something's wrong! Aborting..."
            T = template_dict["T_counts"][4]  # get counts template of GCE
        A = T_counts_rescale_fac[i_temp]
        best_fit_maps_templates[i_temp, :, :] = np.random.poisson((A * T)[None], size=(n_maps, n_pix_ROI))

    # Now, do the Non-Poissonian maps
    tau_index = (fermi_pred["gce_hist"].shape[0] - 1) // 2  # which quantile level tau to use? MEDIAN

    # Exposure map (uncompressed)
    fermi_exp = get_template(fermi_folder, "exp")
    fermi_exp = hp.reorder(fermi_exp, r2n=True)
    mean_exp = np.mean(fermi_exp)
    fermi_rescale = fermi_exp / mean_exp

    # Get Fermi PSF
    PSF = get_Fermi_PDF_sampler()

    # Initialise arrays
    flux_arrays_all = []
    maps_PS_all = []

    # Interpolate Fermi prediction such that fluxes can lie anywhere between the bin centres?
    DO_INTERP = False

    # Deactivate Poissonian scatter for number of PSs?
    DEACTIVATE_POISSON_N_PS = False
    int_fun = trapz  # trapz, simps, ...

    for i_temp, temp in enumerate(T_NP):
        print("Starting with Non-Poissonian model", temp)
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

        # relative F dN/dF -> dN/dF with correct scaling
        F_ary = 10 ** bin_centres
        F_tot_template = FFs[i_temp + len(T_P)] * total_fermi_flux
        F_tot_template_entire_map = F_tot_template * area_template_frac
        temp_ind_hist = 0 if "gce" in temp else 1
        hist_rel = fermi_pred["gce_hist"][tau_index, :, temp_ind_hist]

        # The histogram is normalised such that sum(histogram_rel) = 1
        # sum(histogram_abs) = F_tot, where histogram_rel = histogram_abs / F_tot
        hist_abs = hist_rel * F_tot_template_entire_map

        # Scale such that integral over log10(F) gives F_tot, rather than the sum
        int_hist_abs = int_fun(hist_abs, bin_centres)
        FdNdlogF = hist_abs / int_hist_abs * F_tot_template_entire_map
        # now, int_fun(FdNdlogF, bin_centres) = F_tot_template_entire_map
        print("Integral over FdNdlogF dlogF:", int_fun(FdNdlogF, bin_centres),
              "\nTotal flux of template:", F_tot_template_entire_map)

        if DO_INTERP:
            F_ary_interp = np.logspace(bin_centres[0], bin_centres[-1], 1000000)
            FdNdlogF_interp = sp.interpolate.interp1d(np.log10(F_ary), FdNdlogF, kind="linear")(np.log10(F_ary_interp))
        else:
            F_ary_interp = F_ary
            FdNdlogF_interp = FdNdlogF

        # From FdNdlogF, compute number of expected PSs via
        #  N_exp = int dN/dlogF dlogF
        N_tot_template = int_fun(FdNdlogF / F_ary, bin_centres)
        print("Expected number of PSs:", N_tot_template)

        # Sample n_maps times from a Poisson distribution to obtain the number of PSs in the map
        if DEACTIVATE_POISSON_N_PS:
            N_PS = np.repeat(int(np.round(N_tot_template)), n_maps)  # Turn off Poisson scatter
        else:
            N_PS = np.random.poisson(N_tot_template, size=n_maps)  # Poisson scatter in the number of PSs

        # Sample the fluxes
        # We have log-spaced bins, so the PDF is dN/d(log F) = F dN/dF !!!
        flux_array = [PDFSampler(F_ary_interp, FdNdlogF_interp / F_ary_interp)(N_PS[i]) for i in range(n_maps)]

        avg_tot_flux = np.mean([f.sum() for f in flux_array])
        print("Total flux (avg. over the maps):", avg_tot_flux)
        avg_num_PS = np.mean([len(f) for f in flux_array])
        print("Mean number of PSs:", avg_num_PS)

        maps_PS, n_phot_PS = [], []

        # Generate the maps
        # NOTE: This function outputs COMPRESSED MAPS!!!
        for i in tqdm.tqdm(range(n_maps)):
            map_ = run(np.asarray(flux_array[i]), params["indexes"][0], T_final, fermi_exp, PSF, "",
                                 save=False, upscale_nside=16384, verbose=False, is_nest=True)
            maps_PS.append(map_)

        maps_PS = np.asarray(maps_PS)
        print("Mean number of counts in the PS maps:", maps_PS[:, :].sum(1).mean())
        print("Estimated number of counts in the Fermi map from this template:",
              count_fracs[i_temp + len(T_P)] * total_fermi_counts, "\n")

        flux_arrays_all.append(flux_array)
        maps_PS_all.append(maps_PS)

    maps_PS_all = np.asarray(maps_PS_all)  # Shape: n_templates x n_maps x n_pixels

    # Initialise output dict
    best_fit_data = dict()
    best_fit_data["gce_hist"] = np.zeros((n_maps, len(bin_centres), len(T_NP)))

    # Insert in best_fit_maps_templates
    for i_temp in range(len(T_NP)):
        best_fit_maps_templates[i_temp + len(T_P), :, :] = maps_PS_all[i_temp]

    # Set the final map: Poisson + PS
    final_map_counts = best_fit_maps_templates.sum(0)
    # remove exposure correction !!!!
    # NOTE: When getting data using generator, this is done within generator!
    best_fit_data["data"] = final_map_counts / template_dict["rescale"][None]

    print("Avg. number of counts:", final_map_counts.sum(1).mean())
    print("Counts in the Fermi map:", total_fermi_counts)

    # Set the labels
    flux_maps_templates = best_fit_maps_templates / np.expand_dims(exp_compressed_nest, [0, 1])
    flux_tot_templates = flux_maps_templates.sum(2)
    FF_templates = flux_tot_templates / flux_tot_templates.sum(0, keepdims=True)
    print("Mean FFs of simulated maps:", FF_templates.mean(1))
    print("FFs in the Fermi map:      ", FFs)
    best_fit_data["label"] = FF_templates.T

    # Compute the histograms
    # NOTE: THE HISTOGRAMS HERE ARE COMPUTED USING THE FLUX ARRAY THAT BELONGS TO ALL THE COUNTS, ALSO THOSE THAT DO
    # NOT LIE IN THE ROI!
    # However, this should not lead to a bias but just to increased scatter, as the dNdF is spatially uniform.
    F_power = 1.0
    n_hists = 1 if SMOOTH_GCE else 2
    hists_all = np.zeros((n_maps, len(bin_centres), n_hists))
    for i_temp in range(len(T_NP)):
        hist_input = flux_arrays_all[i_temp]
        dNdF_hist = np.asarray([np.histogram(hist_input[i], weights=hist_input[i] ** F_power, bins=bin_edges)[0]
                                for i in range(n_maps)])
        dNdF_hist_sum = dNdF_hist.sum(1)
        hists_all[:, :, i_temp] = dNdF_hist / np.expand_dims(dNdF_hist_sum, -1)
    best_fit_data["gce_hist"] = hists_all

    bar_width = np.diff(bin_centres)[0]
    for i_temp, temp in enumerate(T_NP):
        temp_ind_hist = 0 if "gce" in temp else 1
        print("Template", temp, "\n")
        print("Mean histogram for simulated maps:", hists_all[:, :, i_temp].mean(0))
        print("Fermi map histogram              :", fermi_pred["gce_hist"][tau_index, :, temp_ind_hist])
        print("\n")

        plt.figure()
        plt.bar(bin_centres, fermi_pred["gce_hist"][tau_index, :, temp_ind_hist], color="k", alpha=0.5, width=bar_width)
        plt.bar(bin_centres, hists_all[:, :, i_temp].mean(0), color="b", alpha=0.5, width=bar_width)

    # Save
    np.save(os.path.join(model.get_path("checkpoints"), data_out_file), best_fit_data)

    # # TO CHECK IF EVERYTHING MAKES SENSE:
    # # set average of best fit maps to be Fermi map, then go through everything.
    # # -> should give the same best fit maps as for the real Fermi maps ("idempotent operation")
    # fermi_pred["logits_mean"] = np.tile(best_fit_data["labels"].mean(0, keepdims=True), [len(tau_vec), 1])
    # fermi_pred["gce_hist"] = np.tile(best_fit_data["gce_hist"].mean(0, keepdims=True), [len(tau_vec), 1, 1])


# # quickly evaluate the median
# median_pred = model.predict({"data": best_fit_data["data"][:1]}, None, tau_hist=0.5 * np.ones((n_maps, 1)))
# print("FFs:                             ", FFs)
# print("NN median prediction for MC maps:", np.median(median_pred["logits_mean"], 0))
# for i_temp in range(1):
#     plt.figure()
#     plt.bar(bin_centres, fermi_pred["gce_hist"][tau_index, :, i_temp], color="k", alpha=0.5, width=bar_width)
#     plt.bar(bin_centres, np.median(median_pred["gce_hist"], 0)[:, i_temp], color="b", alpha=0.5, width=bar_width)
