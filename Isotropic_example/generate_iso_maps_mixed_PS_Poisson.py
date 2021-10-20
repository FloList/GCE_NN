"""
This script generates isotropic maps consisting of Poisson + PS flux. The predicted SCD histograms for these maps
can be used as training data for the NN that constrains the Poisson flux fraction.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import sys
from gce_utils import *
from deepsphere_GCE_workflow import build_NN
import scipy as sp
import random
import os
import copy
import seaborn as sns
import tqdm
from NPTFit import create_mask as cm  # Module for creating masks
sns.set_style("ticks")
sns.set_context("talk")
plt.ion()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'rocket'
# ########################################################
print("\n\nFound TF", tf.__version__, ".")
tf.compat.v1.disable_eager_execution()


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


def get_PS_flux_array(skew_, loc_, scale_, flux_lims_, max_total_flux, size_approx_mean=10000, flux_log=False):
    # Draw the desired flux
    if flux_log:
        flux_desired = 10 ** np.random.uniform(*flux_lims_)
    else:
        flux_desired = np.random.uniform(*flux_lims_)
    # Calculate the expected value of 10^X
    exp_value = (10 ** sp.stats.skewnorm.rvs(skew_, loc=loc_, scale=scale_, size=size_approx_mean)).mean()
    # Determine the number of sources (minimum: 1)
    n_sources = max(1, int(np.round(flux_desired / exp_value)))
    # Initialise total flux
    tot_flux = np.infty

    # Draw fluxes until total flux is in valid range
    while tot_flux >= max_total_flux:
        flux_arr_ = 10 ** sp.stats.skewnorm.rvs(skew_, loc=loc_, scale=scale_, size=n_sources)
        tot_flux = flux_arr_.sum()
        # If total flux > max-total_flux: reduce n_sources
        if tot_flux > max_total_flux:
            n_sources = int(max(1, n_sources // 1.05))

    return flux_arr_


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

    NO_PSF = True  # deactivate PSF?
    DO_FAINT = True  # take the NN trained on fainter maps (NO_PSF = True only)?

    if NO_PSF:
        if DO_FAINT:
            TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_add_two_faint_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
        else:
            TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_add_two_no_PSF_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)
    else:
        if DO_FAINT:
            raise NotImplementedError
        TEST_EXP_PATH = "./checkpoints/Iso_maps_combined_add_two_IN_bs_256_softplus_pre_gen"  # if not None: load the specified NN (otherwise: parameters_*.py)

    test_folder = None  # (CNN only)
    models_test = None  # (CNN only)
    parameter_filename = None  # "parameters_CNN_pre_gen"
    # ########################################################

    n_chunks = 5  # Number of chunks to compute for each process
    n_maps = 2048  # Number of maps per chunk

    if GADI:
        try:
            JOB_ID = sys.argv[1]
        except IndexError:
            print("NO JOB_ID provided! Setting JOB_ID = 0!")
            JOB_ID = 0
    else:
        JOB_ID = 0
    print("JOB ID is", JOB_ID, ".\n")

    # Set a random seed for numpy (using random, because numpy duplicates random number generator for multiple processes)
    random_seed = random.randint(0, int(2 ** 32 - 1))
    np.random.seed(random_seed)
    print("Job ID:", JOB_ID, "Random Seed:", random_seed)

    # Define settings for the single GCE PS population
    prior_dict = dict()
    mean_exp = [-2, 1.5] if DO_FAINT else [-1, 1.5]
    prior_dict["iso_PS"] = {"mean_exp": mean_exp, "var_exp": 0.1, "skew_std": 3.0, "flux_lims": [1, 50000],
                            "flux_log": False}

    # Build model
    model, params, input_test, input_test_db, generator_test, ds_test, fermi_counts \
        = build_NN(NN_TYPE, GADI, DEBUG, TASK, TEST_EXP_PATH, test_folder, models_test, PRE_GEN,
                   parameter_filename)

    bin_edges = copy.copy(params["gce_hist_bins"])
    data_out_folder = "Mixed_PS_Poisson"
    data_out_file = "Maps_"
    nside = params["nside"]
    mkdir_p(os.path.join(model.get_path("checkpoints"), data_out_folder))

    if GADI:
        sys.path.append('/scratch/u95/fl9575/GCE_v2/Isotropic_example/')
    from ps_mc_fast_entire_sky_restricted import run

    # Bins to plot: np.log10(bins) for dNdF histogram, else: bins
    do_log = True
    bins_to_plot = np.log10(bin_edges) if do_log else bin_edges
    bins_to_plot[0] = 2 * bins_to_plot[1] - bins_to_plot[2]
    bins_to_plot[-1] = 2 * bins_to_plot[-2] - bins_to_plot[-3]
    bin_centres = (bins_to_plot[1:] + bins_to_plot[:-1]) / 2.0

    # Get Fermi PSF
    if NO_PSF:
        PSF = None
    else:
        PSF = get_Fermi_PDF_sampler()

    n_pix_ROI = generator_test.settings_dict["unmasked_pix"].shape[0]

    # Iterate over the chunks
    for i_chunk in range(n_chunks):
        print("STARTING WITH CHUNK", i_chunk)

        # Initialise mixed maps, length: 2
        mixed_maps_templates = np.zeros((len(params["models"]) + 1, n_maps, n_pix_ROI))
        print("Starting with Poissonian emission")
        T = np.ones(n_pix_ROI)
        A = np.random.uniform(0, 50000 / n_pix_ROI, size=(n_maps,))  # A UNIFORMLY DRAWN BETWEEN [0, 50,000] in the entire map
        mixed_maps_templates[0, :, :] = np.asarray([np.random.poisson(A[i] * T) for i in range(n_maps)])

        # Now, do the Non-Poissonian maps
        # Initialise arrays
        flux_arrays_all = []
        maps_PS_all = []

        print("Starting with Non-Poissonian emission")

        # Define mask
        total_mask_neg = np.ones(hp.nside2npix(params["nside"]))
        total_mask_neg[params["indexes"][0]] = 0

        T_uncompressed = np.ones(hp.nside2npix(params["nside"]))
        indices_ROI = params["indexes"][0]

        # if without PSF: no leaking into / out of ROI: can directly take masked ROI
        if NO_PSF:
            T_masked = T_uncompressed * (1 - total_mask_neg)
            T_final = T_masked / T_masked.sum()

        # if with PSF: Do NOT apply mask, rather rescale!
        else:
            T_final = T_uncompressed / T_uncompressed.sum()

        # Template needs to be normalised to sum up to unity for the new implementation!
        # Might need to do this twice because of rounding errors
        while T_final.sum() > 1.0:
            T_final /= T_final.sum()
        if T_final.sum() != 1.0:
            print("WARNING: TEMPLATE SUM IS NOT EXACTLY 1 BUT", T_final.sum(), "!")

        # if with PSF: correct for smaller ROI
        if NO_PSF:
            area_template_frac = 1.0
        else:
            T_final_masked = T_final * (1 - total_mask_neg)
            area_template_frac = 1.0 / T_final_masked.sum()

        F_ary = 10 ** bin_centres

        # Draw the SCD parameters
        temp = "iso_PS"
        mean_draw = np.random.uniform(*prior_dict[temp]["mean_exp"], size=n_maps)
        var_draw = prior_dict[temp]["var_exp"] * np.random.chisquare(1, size=n_maps)
        skew_draw = np.random.normal(loc=0, scale=prior_dict[temp]["skew_std"], size=n_maps)
        flux_lims = np.asarray(prior_dict[temp]["flux_lims"]) * area_template_frac

        flux_array = [get_PS_flux_array(skew_draw[i], mean_draw[i], np.sqrt(var_draw[i]),
                                           flux_lims, flux_lims[1], flux_log=prior_dict[temp]["flux_log"])
                      for i in range(n_maps)]

        avg_tot_flux = np.mean([f.sum() for f in flux_array])
        print("Total flux (avg. over the maps):", avg_tot_flux)
        avg_num_PS = np.mean([len(f) for f in flux_array])
        print("Mean number of PSs:", avg_num_PS)

        maps_PS = []

        # Generate the maps
        # NOTE: This function outputs COMPRESSED MAPS!!!
        for i in tqdm.tqdm(range(n_maps)):
            map_ = run(np.asarray(flux_array[i]), indices_ROI, T_final, np.ones_like(T_final), PSF, "",
                                 save=False, upscale_nside=16384, verbose=False, is_nest=True)
            maps_PS.append(map_)

        maps_PS = np.asarray(maps_PS)
        print("Mean number of counts in the PS maps:", maps_PS[:, :].sum(1).mean())

        # Initialise output dict
        best_fit_data = dict()
        best_fit_data["gce_hist"] = np.zeros((n_maps, len(bin_centres), 1))  # NOTE: histogram only accounts for PS contribution of course!

        # Insert in mixed_maps_templates
        mixed_maps_templates[1, :, :] = maps_PS

        # Now: sort GCE Poissonian / Non-Poissonian counts according to tot. counts in decreasing / increasing order
        #   -> when adding them up, the fraction of Poisson emission will be uniformly distributed rather than triangular dist.!
        count_sum_PS = mixed_maps_templates[1].sum(1)
        count_sum_DM = mixed_maps_templates[0].sum(1)
        count_sum_PS_sort_inds = np.argsort(count_sum_PS)
        count_sum_DM_sort_inds = np.argsort(count_sum_DM)
        rand_perm = np.random.permutation(n_maps)  # randomly permute (jointly!)
        count_sum_PS_sort_inds = count_sum_PS_sort_inds[rand_perm]
        count_sum_DM_sort_inds = count_sum_DM_sort_inds[::-1][rand_perm]  # reverse order for Poisson
        mixed_maps_templates[0, :, :] = mixed_maps_templates[0, count_sum_DM_sort_inds, :]
        mixed_maps_templates[1, :, :] = mixed_maps_templates[1, count_sum_PS_sort_inds, :]
        # Also need to sort PS flux arrays for GCE!
        flux_array = np.asarray(flux_array)[count_sum_PS_sort_inds].tolist()

        # Compute P/(P + NP) ratio
        Poiss_count_frac = mixed_maps_templates[0, :, :].sum(1) / \
                           (mixed_maps_templates[0, :, :].sum(1) + mixed_maps_templates[1, :, :].sum(1))
        if i_chunk == 0 and JOB_ID == 0:
            fig_P_frac, ax_P_frac = plt.subplots(1, 1)
            ax_P_frac.hist(Poiss_count_frac)
            fig_P_frac.savefig(os.path.join(model.get_path("checkpoints"), data_out_folder,
                                            data_out_file + "Poiss_frac_hist.pdf"), bbox_inches="tight")
            plt.close("all")

        # Set the final map
        final_map_counts = mixed_maps_templates.sum(0)
        best_fit_data["data"] = final_map_counts

        print("Avg. number of counts:", final_map_counts.sum(1).mean())

        # Set the FF labels: trivial (only iso template)
        best_fit_data["label"] = np.ones((n_maps, 1))

        # Compute the Poissonian fraction within the GCE (same as GCE_Poiss_count_frac, but after dividing by exposure -> flux)
        best_fit_data["gce_poiss_ff"] = np.expand_dims(Poiss_count_frac, -1)

        # Compute the histograms (for GCE: only PS component)
        # NOTE: The histograms are computed using the flux array that belongs to all the counts, also those that do not
        # lie in the ROI.
        # However, this should not lead to bias but just to increased scatter, as the dNdF is spatially uniform.
        # Also, the true histogram labels are never really used anywhere...
        F_power = 1.0
        n_hists = 1
        hists_all = np.zeros((n_maps, len(bin_centres), 1))
        hist_input = flux_array
        dNdF_hist = np.asarray([np.histogram(hist_input[i], weights=hist_input[i] ** F_power, bins=bin_edges)[0]
                                for i in range(n_maps)])
        dNdF_hist_sum = dNdF_hist.sum(1)
        hists_all[:, :, 0] = dNdF_hist / np.expand_dims(dNdF_hist_sum, -1)
        best_fit_data["gce_hist"] = hists_all

        # Save
        np.save(os.path.join(model.get_path("checkpoints"), data_out_folder, data_out_file
                                          + str(JOB_ID) + "_" + str(i_chunk)), best_fit_data)
        print("Data saved for JOB_ID {:}, chunk {:}!".format(JOB_ID, i_chunk))
