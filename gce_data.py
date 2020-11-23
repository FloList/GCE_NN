import tensorflow as tf
import numpy as np
import healpy as hp
import os
import pickle
import random
from gce_utils import remove_exposure_correction, masked_to_full, healpix_to_cart, dict_to_array, get_template, \
     split_into_N_and_S, flatten_var_fracs, dnds
from NPTFit import create_mask as cm  # Module for creating masks
from sklearn.model_selection import train_test_split
from copy import copy
import psutil
import ray

# # # # # # # # # # GCE Data classes # # # # # # # # # #
class GCEData:
    def __init__(self, data: tf.Tensor, label: tf.Tensor, var_fracs=None, extra_info=None, gce_hist=None):
        """
        Default class for GCE Tensorflow data
        :param data: map of photon counts (partial or full) (possibly rescaled such that the exposure correction is removed)
        :param label: flux fraction for each template
        :param var_fracs: fraction of each variant for each template
        :param extra_info: additional info (that is not needed for training the NN, but might be for analysis)
        """
        self.data = data
        self.label = label
        self.extra_info = extra_info
        if var_fracs is not None:
            self.var_fracs = var_fracs
        if gce_hist is not None:
            self.gce_hist = gce_hist

    def vars(self):
        """
        :return: dictionary with data attributes that are not None
        """
        all_vars = vars(self)
        set_keys = [v != None for v in all_vars.values()]
        return {k: all_vars[k] for k in np.asarray(list(all_vars.keys()))[set_keys]}

# # # # # # # # # # PairGenerator classes # # # # # # # # # #
class PairGenerator(object):
    """Base PairGenerator class"""

    def __init__(self, models, nside=128, fill_value=0.0, test=False, indices=None):
        self.models = copy(models)
        self.nside = nside
        self.fill_value = fill_value
        self.indices = copy(indices)
        self.is_training = not test


# PairGenerator class for generating data on-the-fly
class PairGeneratorOnTheFly(PairGenerator):
    def __init__(self, template_folder, models, model_vars, prior_dict, log_priors=True, nside=128, fill_value=0.0, test=False,
                 indices=None, remove_exp=True, project_2D=None, ROI_ring=None, map_cond=None, flux_cond=None,
                 n_phot_cond=None, p_cond=None, mask=None, const_exp=False, folder_maps_PS="", test_fraction_PS=0.2,
                 combine_gce=False, gce_PS_var=None):
        """
        Initialise the pair generator for on-the-fly data generation
        :param template_folder: folder with input templates
        :param models: list of models to be used as labels, e.g. ["gce", "iso", "dif", ...]
        :param model_vars: list of lists: for each model, all the variants that shall be used (e.g. "dif_O_pibs", "dif_A_pibs", ...)
        :param prior_dict: dictionary with (uniform) prior limits for each template
        :param log_priors: specify log10(A) instead of A
        :param nside: nside for the data set
        :param fill_value: value to fill masked pixels, default: 0.0
        :param test: set this to False for training generator and True for validation/test generator
        :param indices: input data needs to be extended to match size of indices for each
                        convolutional layer (necessary to ensure divisibility by 4^p when pooling)
        :param remove_exp: remove exposure correction
        :param project_2D: list with 2 elements for the x/y resolution for projecting data to Cartesian 2D image (for 2D CNN)
        :param ROI_ring: ROI outer radius (needed when projecting to 2D image)
        :param map_cond: condition on the map that is enforced with a prob. of p_cond
        :param flux_cond: condition on the flux fractions that is enforced with a prob. of p_cond
        :param n_phot_cond: condition on the list of photon counts per PS that is enforced with a prob. of p_cond
        :param p_cond: probability for enforcing the conditions on the map / flux fractions
        :param mask: if not None: masked pixels within ROI in RING format
        :param const_exp: take constant Fermi mean exposure for map generation
        :param folder_maps_PS: if PSs shall be added (must be pre-generated): folder name
        :param test_fraction_PS: fraction of PS maps to be used for testing
        :param combine_gce: GCE DM + PS will be added and treated as a single template
        :param gce_PS_var: name of GCE PS variant
        """
        super(PairGeneratorOnTheFly, self).__init__(models, nside, fill_value, test, indices)
        self.model_vars = copy(model_vars)
        self.combine_gce = combine_gce
        if self.combine_gce:
            self.gce_index = np.argwhere(["gce" in name for name in self.models]).flatten()
            assert len(self.gce_index) <= 1, "Only 1 GCE template (DM) is allowed when 'combine_gce' is on!"
            assert gce_PS_var is not None and type(gce_PS_var) == list, "gce_PS_var must be specified as a list!"
            self.models.append("gce_PS_for_combining")
            self.model_vars.append(gce_PS_var)
        self.prior_dict = prior_dict
        self.log_priors = log_priors
        self.remove_exp = remove_exp
        self.const_exp = const_exp
        self.project_2D = project_2D
        self.ROI_ring = ROI_ring
        self.map_cond = map_cond
        self.flux_cond = flux_cond
        self.n_phot_cond = n_phot_cond
        self.p_cond = p_cond
        self.cond_must_be_imposed = False
        self.mask_fct = lambda x: x[indices[0]]  # assumes NEST format!
        self.unmask_fct = lambda x, nside_loc=nside: masked_to_full(x, unmasked_pix=indices[0], nside=nside_loc)  # assumes NEST format!
        if mask is None:
            mask = np.zeros(hp.nside2npix(nside)).astype(bool)
        self.mask = mask
        self.folder_maps_PS = folder_maps_PS
        self.nside_fermi = 128
        self.NP_indices = np.argwhere([np.any(["_PS" in var for var in vars]) for vars in self.model_vars]).flatten()
        self.P_indices = np.setdiff1d(range(len(self.models)), self.NP_indices)

        # PS-specific attributes
        self.file_no = dict()
        self.index_in_array = dict()
        self.active_file = dict()
        self.data_dict = dict()
        self.unmasked_pix_PS = dict()
        self.nside_PS = dict()
        self.train_files_PS, self.test_files_PS = dict(), dict()
        self.pixel_mapping_PS_set = dict()
        self.pixel_mapping_PS_get = dict()

        # Exposure
        # Fermi exposure (possibly needs to be removed from the templates)
        self.fermi_exp = get_template(template_folder, "exp")  # in RING format
        self.fermi_mean_exp = self.fermi_exp.mean()
        self.fermi_rescale = self.fermi_exp / self.fermi_mean_exp

        # Flux to count conversions
        self.fermi_rescale_to_apply = self.mask_fct(hp.reorder(hp.ud_grade(self.fermi_rescale, nside_out=nside), r2n=True))
        if const_exp:
            self.rescale_to_apply = np.ones_like(self.fermi_rescale_to_apply)
        else:
            self.rescale_to_apply = self.fermi_rescale_to_apply

        # Get templates (exposure-corrected!) and prepare loading of PS maps
        template_maps_raw = dict()  # in RING format
        template_maps = dict()  # in NEST format, masked, and only for the required pixels
        for i_temp, temp in enumerate(self.models):
            # PREPARE PS TEMPLATES
            if "_PS" in temp:
                for temp_var in self.model_vars[i_temp]:
                    # Prepare PS templates
                    folder_PS = os.path.join(folder_maps_PS, temp_var)

                    if not test:
                        all_files = os.listdir(folder_PS)
                        all_files = [file for file in all_files if "EXCLUDE" not in file]  # Don't include files containing "EXCLUDE"
                        try:
                            settings_ind = np.argwhere(["settings" in file for file in all_files])[0][0]
                            settings_file = open(os.path.join(folder_PS, all_files[settings_ind]), 'rb')
                            settings_dict = pickle.load(settings_file)
                            settings_file.close()
                        except (FileNotFoundError, EOFError, IOError):
                            print("Opening settings file failed. Aborting...")
                            os._exit(1)

                        if len(np.unique(settings_dict["rescale"])) == 1 and len(np.unique(self.rescale_to_apply)) != 1:
                            print("It seems like the exposure for the PSs is uniform while that for the Poissonian templates is non-uniform. Aborting!")
                            os._exit(3)
                        if len(np.unique(settings_dict["rescale"])) != 1 and len(np.unique(self.rescale_to_apply)) == 1:
                            print("It seems like the exposure for the PSs is non-uniform while that for the Poissonian templates is uniform. Aborting!")
                            os._exit(4)
                        self.unmasked_pix_PS[temp_var] = settings_dict["unmasked_pix"]
                        nside_PS = hp.npix2nside(len(settings_dict["T_corr"]))

                        # if nside of PS matches: already determine here the index mapping from "self.unmasked_pix_PS" to "indices"
                        if nside_PS == nside:
                            indices_in = np.arange(len(self.unmasked_pix_PS[temp_var]))
                            indices_in_ring_full = masked_to_full(indices_in, self.unmasked_pix_PS[temp_var], nside=self.nside, fill_value=np.nan)
                            indices_in_ring_full[self.mask.astype(bool)] = np.nan
                            indices_out = self.mask_fct(hp.reorder(indices_in_ring_full, r2n=True))
                            indices_out_relevant = indices_out[np.argwhere(np.logical_not(np.isnan(indices_out))).flatten()].astype(int)
                            self.pixel_mapping_PS_get[temp_var] = [np.argwhere(ind_out == indices_in).flatten()[0] for ind_out in indices_out_relevant]
                            self.pixel_mapping_PS_set[temp_var] = np.logical_not(np.isnan(indices_out))

                        data_files = np.asarray(all_files)[np.setdiff1d(range(len(all_files)), settings_ind)]
                        # for train-test split: fix random state to 0 to always get the same split!
                        self.train_files_PS[temp_var], self.test_files_PS[temp_var] = train_test_split(data_files, test_size=test_fraction_PS, random_state=0)
                        print(self.test_files_PS[temp_var])

                    self.file_no[temp_var] = 0
                    self.index_in_array[temp_var] = 0
                    self.active_file[temp_var] = []

                    if not test:
                        self.nside_PS[temp_var] = hp.npix2nside(len(settings_dict["T_corr"]))
                        if self.nside_PS[temp_var] != self.nside:
                            print("\nWARNING! N_SIDE parameter of PS template", temp_var, "=", self.nside_PS[temp_var],
                                  "whereas a resolution with N_SIDE =", self.nside, "is chosen!\n")

            # PREPARE POISSONIAN TEMPLATES
            else:
                for temp_var in self.model_vars[i_temp]:
                    template_maps_raw[temp_var] = get_template(template_folder, temp_var)
                    if const_exp:  # if maps shall be generated with spatially homogeneous exposure: remove exposure correction
                        template_maps_raw[temp_var] = remove_exposure_correction(template_maps_raw[temp_var], self.fermi_rescale)
                    if nside != 128:
                        template_maps[temp_var] = self.mask_fct(hp.reorder(hp.ud_grade(template_maps_raw[temp_var], nside_out=nside) * (1 - mask), r2n=True))
                    else:
                        template_maps[temp_var] = self.mask_fct(hp.reorder(template_maps_raw[temp_var] * (1 - mask), r2n=True))
        self.template_maps = template_maps
        if self.template_maps:
            self.input_length = len(self.template_maps[self.model_vars[0][0]])
        elif self.pixel_mapping_PS_set:
            self.input_length = len(self.pixel_mapping_PS_set[self.model_vars[0][0]])
        else:
            raise RuntimeError("Neither Poissonian nor non-Poissonian templates found! Aborting...")


    def copy_generator_attributes(self, test_filenames, unmasked_pix_PS, nside_PS, pixel_mapping_PS_set,
                                  pixel_mapping_PS_get):
        self.test_files_PS = test_filenames
        self.unmasked_pix_PS = unmasked_pix_PS
        self.nside_PS = nside_PS
        self.pixel_mapping_PS_set = pixel_mapping_PS_set
        self.pixel_mapping_PS_get = pixel_mapping_PS_get

# CNN: PairGenerator class for generating data on-the-fly
class PairGeneratorCNNOnTheFly(PairGeneratorOnTheFly):
    def __init__(self, map_cond=None, flux_cond=None, n_phot_cond=None, p_cond=None, gce_return_hist=False, gce_hist_bins=None,
                 which_hist=1, power_of_F=1, no_PSF=False, *args, **kwargs):
        """
        Initialise the pair generator for the CNN architecture: data is generated ON-THE-FLY!
        :param map_cond: if not None: with a probability of p_cond, impose condition on map (lambda)
        :param flux_cond: if not None: with a probability of p_cond, impose condition on flux contributions (lambda)
        :param n_phot_cond: if not None: with a probability of p_cond, impose condition on photon list per PS (lambda)
        :param p_cond: probability that conditions are enforced (between 0 and 1)
        :param gce_return_hist: if True: return the histogram of the GCE DM + PS distribution
        :param gce_hist_bins: bins for the histogram
        :param which_hist: flag for histogram data to use (see parameters_CNN.py)
        :param power_of_F: if which_hist = 1: histogram of dN/dF * F^gamma, where gamma = power_of_F
        :param no_PSF: if True: use 2nd channel that contains the maps WITHOUT PSF smoothing
        """
        super(PairGeneratorCNNOnTheFly, self).__init__(*args, **kwargs)
        self.map_cond = map_cond
        self.flux_cond = flux_cond
        self.p_cond = p_cond
        self.n_phot_cond = n_phot_cond
        self.cond_must_be_imposed = False
        self.gce_return_hist = gce_return_hist
        self.no_PSF = no_PSF
        if gce_return_hist:
            assert self.combine_gce, "gce_return_hist requires combine_gce == True!"
            self.gce_hist_bins = gce_hist_bins
            self.which_hist = which_hist
            self.power_of_F = power_of_F
            assert not self.remove_exp or self.which_hist < 2, "When estimating histograms in terms of counts, exposure correction should NOT be removed!"
            assert self.remove_exp or self.which_hist > 1, "When estimating histograms in terms of counts, exposure correction should NOT be removed!"

    def get_next_pair(self, extra_info=False):
        """
        :return: next data - label pair (if EOF is reach: load from new file, if all files have been used: repeat)
        """
        while True:
            filenames = self.train_files_PS if self.is_training else self.test_files_PS
            ready_to_yield = False
            train_str = "Training:" if self.is_training else "Validation:"
            while not ready_to_yield:
                # Initialise
                template_maps_counts = np.zeros((self.input_length, len(self.models)))
                A = np.zeros(len(self.models))
                var_weights = [None] * len(self.models)

                # Poissonian: draw the log amplitude and generate Poissonian sample
                for i_temp in self.P_indices:
                    temp = self.models[i_temp]
                    # while template_maps_counts[:, i_temp].sum() == 0:  # require that there is at least a single count for each template?
                    # Draw template normalisation
                    A[i_temp] = random.uniform(self.prior_dict[temp][0], self.prior_dict[temp][1])
                    # Draw weighting of the variants randomly from a flat Dirichlet distribution
                    n_vars = len(self.model_vars[i_temp])
                    var_weights[i_temp] = np.random.dirichlet(np.ones(n_vars), size=1).flatten()
                    this_template_map = np.expand_dims(var_weights[i_temp], 0) @ np.asarray([self.template_maps[temp_var] for temp_var in self.model_vars[i_temp]])
                    if np.isnan(A[i_temp]):
                        template_maps_counts[:, i_temp] = np.zeros_like(this_template_map)
                    else:
                        if self.log_priors:
                            template_maps_counts[:, i_temp] = np.random.poisson((10.0 ** A[i_temp]) * this_template_map)
                        else:
                            template_maps_counts[:, i_temp] = np.random.poisson(A[i_temp] * this_template_map)

                # PS: load the sample
                for i_temp in self.NP_indices:
                    assert len(self.model_vars[i_temp]) == 1, "Only 1 variant for each PS template is supported!"
                    temp_var = self.model_vars[i_temp][0]

                    # Get first file
                    if temp_var in self.data_dict.keys():
                        n_maps_per_file = self.data_dict[temp_var]["data"].shape[1] if len(self.data_dict[temp_var]["data"].shape) == 2 else self.data_dict[temp_var]["data"].shape[0]
                    else:
                        n_maps_per_file = 0
                    if len(self.active_file[temp_var]) == 0:
                        self.active_file[temp_var] = filenames[temp_var][self.file_no[temp_var]]
                        self.index_in_array[temp_var] = 0
                        print(train_str, "Opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                        try:
                            data_file = open(os.path.join(self.folder_maps_PS, temp_var, self.active_file[temp_var]), 'rb')
                            self.data_dict[temp_var] = pickle.load(data_file)
                            data_file.close()
                        except (FileNotFoundError, EOFError, IOError):
                            print(train_str, "Error when opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                            os._exit(2)

                    # If end of a file
                    elif self.index_in_array[temp_var] == n_maps_per_file:
                        self.index_in_array[temp_var] = 0
                        self.file_no[temp_var] += 1
                        # If through all files: start at file 0
                        if self.file_no[temp_var] >= len(filenames[temp_var]):
                            self.file_no[temp_var] = 0
                            print(train_str, "All data from all files has been used for template", temp_var + ".")
                        self.active_file[temp_var] = filenames[temp_var][self.file_no[temp_var]]
                        print(train_str, "Opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                        try:
                            data_file = open(os.path.join(self.folder_maps_PS, temp_var, self.active_file[temp_var]), 'rb')
                            self.data_dict[temp_var] = pickle.load(data_file)
                            data_file.close()
                        except (FileNotFoundError, EOFError, IOError):
                            print(train_str, "Error when opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                            os._exit(2)

                    # Get counts and A
                    # format: either   n_pixels_set x n_maps  (old format)   or   n_maps x n_pixels_set x 2  (without PSF in the 2nd channel)
                    if len(self.data_dict[temp_var]["data"].shape) == 2:
                        counts_raw = self.data_dict[temp_var]["data"][:, self.index_in_array[temp_var]]
                    elif len(self.data_dict[temp_var]["data"].shape) == 3:
                        ind_PS_channel = 1 if self.no_PSF else 0
                        counts_raw = self.data_dict[temp_var]["data"][self.index_in_array[temp_var], :, ind_PS_channel]

                    # if no up-/downsampling is needed: can be done quicker pre-computing the indices ..._get/_set
                    if self.nside_PS[temp_var] == self.nside:
                        template_maps_counts[self.pixel_mapping_PS_set[temp_var], i_temp] = counts_raw[self.pixel_mapping_PS_get[temp_var]]
                        if self.gce_return_hist and "gce" in temp_var:
                            assert template_maps_counts[:, i_temp].sum() == counts_raw.sum(), \
                                "PS data was not generated using the selected ROI, which is required for using the brightness histogram!"
                    else:
                        # Up/-downsample if needed (TAKES LONGER!)
                        counts_ring_full = masked_to_full(counts_raw, self.unmasked_pix_PS[temp_var], nside=self.nside_PS[temp_var])
                        if self.gce_return_hist and "gce" in temp_var:
                            assert counts_ring_full.sum() == counts_raw.sum(), \
                                "PS data was not generated using the selected ROI, which is required for using the brightness histogram!"
                        counts_ring_full = hp.ud_grade(counts_ring_full, nside_out=self.nside, power=-2)
                        counts_ring_full_mask_applied = counts_ring_full * (1 - self.mask)
                        template_maps_counts[:, i_temp] = self.mask_fct(hp.reorder(counts_ring_full_mask_applied, r2n=True))

                    var_weights[i_temp] = np.ones(1)
                    if "A" in self.data_dict[temp_var]["info"].keys():
                        A[i_temp] = self.data_dict[temp_var]["info"]["A"][self.index_in_array[temp_var]]
                    else:
                        A[i_temp] = np.nan

                    # PS histogram
                    if self.models[i_temp] == "gce_PS_for_combining" and self.gce_return_hist:
                        # # # # # #
                        # Legacy dN/dF histogram for broken power laws
                        if self.which_hist == -1:
                            # TODO: PROPER IMPLEMENTATION!!!
                            if self.gce_return_hist:
                                # self.index_in_array[temp_var] += 1
                                F_vec = np.logspace(-11., -8.5, len(self.gce_hist_bins) - 1)
                                # dlog10F_vec = np.log10(F_vec)
                                S_vec = F_vec * self.fermi_mean_exp

                                params_sample = np.concatenate([[self.data_dict[temp_var]["info"]["A_corr"][self.index_in_array[temp_var]]],
                                                               self.data_dict[temp_var]["info"]["n"][self.index_in_array[temp_var]],
                                                               self.data_dict[temp_var]["info"]["S"][self.index_in_array[temp_var]]])
                                dN_dF = dnds(params_sample, S_vec)
                                dN_dF_F2 = dN_dF * F_vec ** 2
                                dN_dF_F2_log = np.log10(dN_dF_F2)
                                clip_below = -22  # -20
                                if np.all(dN_dF_F2_log < clip_below):
                                    hist_GCE = np.zeros_like(dN_dF_F2_log, dtype=np.float32)
                                    hist_GCE[dN_dF_F2_log.argmax()] = 1.0
                                else:
                                    dN_dF_F2_log = np.clip(dN_dF_F2_log, clip_below, np.infty)
                                    dN_dF_F2_log_norm = (dN_dF_F2_log - dN_dF_F2_log.min()) / np.abs(np.sum(dN_dF_F2_log - dN_dF_F2_log.min()))
                                    hist_GCE = dN_dF_F2_log_norm
                                # plt.bar(dlog10F_vec, dN_dF_F2_log_norm, width=0.1, alpha=0.5, lw=2)
                                # plt.plot(F_vec, dN_dF_F2); plt.xscale("log"); plt.yscale("log"); plt.ylim([1e-24, 1e-19])
                                # print(params_sample)
                            # # # # # #

                        # dN/dF histogram for arbitrary dN/dFs as saved in "flux_array"
                        elif self.which_hist == 1:
                            if self.gce_return_hist and "flux_arr" in self.data_dict[temp_var].keys():
                                hist_GCE = np.histogram(self.data_dict[temp_var]["flux_arr"][self.index_in_array[temp_var]],
                                                        weights=self.data_dict[temp_var]["flux_arr"][self.index_in_array[temp_var]] ** self.power_of_F,
                                                        bins=self.gce_hist_bins)[0]
                            elif self.gce_return_hist and "flux_arr" not in self.data_dict[temp_var].keys():
                                raise RuntimeError("Error! GCE PS data does NOT contain lists with flux array! Aborting...")

                        # counts per PS histogram as saved in "n_phot"
                        elif self.which_hist == 2:
                            if self.gce_return_hist and "n_phot" in self.data_dict[temp_var].keys():
                                hist_GCE = np.histogram(self.data_dict[temp_var]["n_phot"][self.index_in_array[temp_var]],
                                                       bins=self.gce_hist_bins,  # weight with the counts themselves
                                                       weights=self.data_dict[temp_var]["n_phot"][self.index_in_array[temp_var]])[0]
                                if self.nside_PS[temp_var] == self.nside:
                                    assert hist_GCE.sum() == template_maps_counts[:, i_temp].sum(), \
                                            "Sum of counts in histogram is " + str(hist_GCE.sum()) + ", but the map contains " \
                                            + str(template_maps_counts[:, i_temp].sum()) + " counts! Aborting..."
                            elif self.gce_return_hist and "n_phot" not in self.data_dict[temp_var].keys():
                                raise RuntimeError("Error! GCE PS data does NOT contain lists with photon counts! Aborting...")

                        # counts per pixel histogram disregarding the PSF as saved in the second channel of "data"
                        elif self.which_hist == 3:
                            if self.gce_return_hist and len(self.data_dict[temp_var]["data"].shape) == 3:
                                hist_GCE = np.histogram(self.data_dict[temp_var]["data"][self.index_in_array[temp_var], :, 1],
                                                       bins=self.gce_hist_bins,  # weight with the counts themselves
                                                       weights=self.data_dict[temp_var]["data"][self.index_in_array[temp_var], :, 1])[0]
                            elif self.gce_return_hist and len(self.data_dict[temp_var]["data"].shape) != 3:
                                raise RuntimeError("Error! GCE PS data does NOT contain second channel with map before PSF application! Aborting...")


                # Combine GCE templates if needed
                if self.combine_gce:
                    # Add GCE DM counts to the lowest bin and normalise the histogram
                    # (only if histogram is in terms of counts)
                    if self.gce_return_hist and not self.which_hist in [-1, 1]:
                        hist_GCE[0] += template_maps_counts[:, self.gce_index[0]].sum()
                        hist_normalisation = max(np.sum(hist_GCE), 1.0)
                        hist_GCE = hist_GCE.astype(np.float64) / hist_normalisation
                    # normalise dN/dF
                    elif self.gce_return_hist and self.which_hist == 1:
                        denominator = 1 if np.sum(hist_GCE) == 0 else np.sum(hist_GCE)
                        hist_GCE = hist_GCE.astype(np.float64) / denominator

                    # Add the counts in the last column (GCE PS) to GCE DM
                    template_maps_counts[:, self.gce_index[0]] += template_maps_counts[:, -1]
                    template_maps_counts = template_maps_counts[:, :-1]
                    var_weights = var_weights[:-1]

                # Remove exposure correction
                if self.remove_exp:
                    template_maps_counts /= np.expand_dims(self.rescale_to_apply, -1)

                # Calculate total input map
                input_map = template_maps_counts.sum(1)

                # Calculate FFs (if exposure correction is removed, this is the fraction of flux rather than the counts)
                labels = template_maps_counts.sum(0) / template_maps_counts.sum()

                # If necessary: map to 2D image for 2D CNN
                if self.project_2D is not None:
                    img = healpix_to_cart(self.unmask_fct(input_map), nside=self.nside,
                                          lonra=[-self.ROI_ring, self.ROI_ring],
                                          latra=[-self.ROI_ring, self.ROI_ring], n_pixels=self.project_2D)
                    input_map = img

                # Check if conditions are satisfied
                ready_to_yield = True
                if self.flux_cond is not None or self.map_cond is not None:
                    impose_cond = self.cond_must_be_imposed or (self.p_cond > np.random.uniform(0, 1, 1))
                    if impose_cond:
                        flux_cond_result = True if self.flux_cond is None else self.flux_cond(labels)
                        map_cond_result = True if self.map_cond is None else self.map_cond(input_map)
                        if self.n_phot_cond is not None:
                            temp_comb = np.argwhere(np.asarray(self.models) == "gce_PS_for_combining").flatten()[0]
                            gce_PS_comb_name = self.model_vars[temp_comb][0]
                            this_hist = self.data_dict[gce_PS_comb_name]["n_phot"][self.index_in_array[gce_PS_comb_name]]
                            n_phot_cond_result = self.n_phot_cond(this_hist)
                        else:
                            n_phot_cond_result = True
                        ready_to_yield = flux_cond_result and map_cond_result and n_phot_cond_result

                # Break
                if ready_to_yield:
                    break
                else:
                    self.cond_must_be_imposed = True
                    for key in self.index_in_array.keys():
                        self.index_in_array[key] += 1

            # Yield the sample
            # print("Yielding file", self.active_file, "index", self.index_in_array)
            output_dict = {"data": input_map.astype(np.float32), "label": labels.astype(np.float32),
                           "var_fracs": flatten_var_fracs(var_weights).astype(np.float32)}
            if np.any(extra_info):
                output_dict["extra_info"] = A.astype(np.float32)
            if self.gce_return_hist:
                output_dict["gce_hist"] = hist_GCE

            yield output_dict

            # Reset
            for key in self.index_in_array.keys():
                self.index_in_array[key] += 1
            self.cond_must_be_imposed = False


# CNN: PairGenerator class for pre-generated data
class PairGeneratorCNNPreGenerated(PairGenerator):
    def __init__(self, data_folder, models, nside=128, unmask=False, fill_value=0.0, test=False, ring2nest=False,
                 indices=None, remove_exp=True, project_2D=None, ROI_ring=None, map_cond=None, flux_cond=None,
                 p_cond=None):
        """
        Initialise the pair generator for CNNs: data is LOADED, not generated on-the-fly!
        :param data_folder: folder with input data (needs to contain subfolders Train (with settings file) and Test)
        :param models: list of models to be used as labels, e.g. ["gce", "iso", "dif", ...]
        :param nside: nside of the data set
        :param unmask: if True: input for DeepSphere will be the entire sphere!
        :param fill_value: value to fill masked pixels (used when unmask=True, but also when unmask=False because image
                           needs to be slightly enlarged to ensure divisibility by 4^p when pooling), default: 0.0
        :param test: set this to False for training generator and True for validation/test generator
        :param ring2nest: if input is saved in RING format, it will then be converted to NEST
        :param indices: if unmask is False: input data needs to be extended to match size of indices for each
                        convolutional layer (necessary to ensure divisibility by 4^p when pooling)
        :param remove_exp: remove exposure correction
        :param project_2D: if not None: list with 2 elements to project data using Cartesian projection for the use of a
                           standard 2D CNN
        :param ROI_ring: if not None: region of interest for 2D projection
        :param map_cond: if not None: with a probability of p_cond, impose condition on map (lambda)
        :param flux_cond: if not None: with a probability of p_cond, impose condition on flux contributions (lambda)
        :param p_cond: probability that conditions are enforced (between 0 and 1)
        """
        super(PairGeneratorCNNPreGenerated, self).__init__(models, nside, fill_value, test, indices)
        folder = os.path.join(data_folder, "Test") if test else os.path.join(data_folder, "Train")
        all_files = os.listdir(folder)
        all_files = [file for file in all_files if "EXCLUDE" not in file]  # Don't include files containing "EXCLUDE"

        # Setting file should be stored with training data
        if not test:
            try:
                settings_ind = np.argwhere(["settings" in file for file in all_files])[0][0]
                settings_file = open(os.path.join(folder, all_files[settings_ind]), 'rb')
                self.settings_dict = pickle.load(settings_file)
                settings_file.close()
            except (FileNotFoundError, EOFError, IOError):
                print("Opening settings file failed. Aborting...")
                os._exit(1)

            # Convert unmasked pix, exposure, and rescale: r2n (data will be converted directly in get_next_pair)
            # NOTE: TEMPLATES ARE NOT CONVERTED SINCE THEY ARE NOT NEEDED!
            if ring2nest:
                self.settings_dict["unmasked_pix_ring"] = self.settings_dict["unmasked_pix"]
                self.settings_dict["unmasked_pix"] = hp.ring2nest(nside, self.settings_dict["unmasked_pix"])
                self.settings_dict["rescale"] = hp.reorder(self.settings_dict["rescale"], r2n=True)
                self.settings_dict["exp"] = hp.reorder(self.settings_dict["exp"], r2n=True)

            if not unmask:
                self.settings_dict["rescale_full"] = self.settings_dict["rescale"]
                self.settings_dict["rescale"] = self.settings_dict["rescale"][indices[0]]
        else:
            settings_ind = -1
            self.settings_dict = []
        self.all_files = [file for (file, i) in zip(all_files, range(len(all_files))) if i != settings_ind]
        self.file_no = 0
        self.index_in_array = 0
        self.active_file = []
        self.data_dict = []
        self.data_folder = folder
        self.do_unmask = unmask
        self.ring2nest = ring2nest
        self.remove_exp = remove_exp
        self.project_2D = project_2D
        self.ROI_ring = ROI_ring
        self.map_cond = map_cond
        self.flux_cond = flux_cond
        self.p_cond = p_cond
        self.cond_must_be_imposed = False

    def get_next_pair(self, verbose=True, extra_info=False):
        """
        :return: next data - label pair (if EOF is reach: load from new file, if all files have been used: repeat)
        """
        while True:
            ready_to_yield = False
            train_str = "Training:" if self.is_training else "Validation:"
            while not ready_to_yield:
                # Get first file
                if len(self.active_file) == 0:
                    self.active_file = self.all_files[self.file_no]
                    self.index_in_array = 0
                    if verbose:
                        print(train_str, "Opening file", self.active_file, "...")
                    try:
                        data_file = open(os.path.join(self.data_folder, self.active_file), 'rb')
                        self.data_dict = pickle.load(data_file)
                        data_file.close()
                    except (FileNotFoundError, EOFError, IOError):
                        print("Opening file", self.active_file, "failed. Aborting...")
                        os._exit(2)

                # If end of a file
                elif self.index_in_array == self.data_dict["data"].shape[1]:
                    self.index_in_array = 0
                    self.file_no += 1
                    # If through all files: start at file 0
                    if self.file_no >= len(self.all_files):
                        self.file_no = 0
                        if verbose:
                            print(train_str, "All PS data from all files for template has been used.")
                    self.active_file = self.all_files[self.file_no]
                    if verbose:
                        print(train_str, "Opening file", self.active_file, "...")
                    try:
                        data_file = open(os.path.join(self.data_folder, self.active_file), 'rb')
                        self.data_dict = pickle.load(data_file)
                        data_file.close()
                    except (FileNotFoundError, EOFError, IOError):
                        print("Opening file", self.active_file, "failed. Aborting...")
                        os._exit(2)

                # Get labels
                labels = np.asarray([self.data_dict["flux_fraction"][key][self.index_in_array] for key in self.models])

                # UNMASK if needed
                # NOTE: if ring2nest, unmasked_pix is already in NESTED format, so no distinction necessary in this case
                if self.do_unmask:
                    new_array = masked_to_full(self.data_dict["data"][:, self.index_in_array],
                                               self.settings_dict["unmasked_pix"],
                                               fill_value=self.fill_value, nside=self.nside)
                else:
                    new_array = self.data_dict["data"][:, self.index_in_array]

                # If masked and in RING format: need to unmask, reorder, and mask again
                if self.ring2nest and not self.do_unmask:
                    full_ring = masked_to_full(new_array, self.settings_dict["unmasked_pix_ring"],
                                               nside=self.nside)  # full array RING
                    full_nest = hp.reorder(full_ring, r2n=True)  # full array NEST
                    new_array = full_nest[self.settings_dict["unmasked_pix"]]  # sparse array new_array NEST

                # If masked and does not cover ROI (given by indices): enlarge!
                if not self.do_unmask and len(self.settings_dict["unmasked_pix"]) != len(self.indices[0]):
                    new_array = \
                    masked_to_full(new_array, self.settings_dict["unmasked_pix"], fill_value=self.fill_value,
                                   nside=self.nside)[self.indices[0]]

                # Remove exposure correction
                if self.remove_exp:
                    new_array = remove_exposure_correction(new_array, self.settings_dict["rescale"])

                # If necessary: map to 2D image for 2D CNN
                if self.project_2D is not None:
                    img = healpix_to_cart(new_array, nside=self.nside, lonra=[-self.ROI_ring, self.ROI_ring],
                                          latra=[-self.ROI_ring, self.ROI_ring], n_pixels=self.project_2D)
                    new_array = img

                # Check if conditions are satisfied
                ready_to_yield = True
                if self.flux_cond is not None or self.map_cond is not None:
                    impose_cond = self.cond_must_be_imposed or (self.p_cond > np.random.uniform(0, 1, 1))
                    if impose_cond:
                        flux_cond_result = True if self.flux_cond is None else self.flux_cond(labels)
                        map_cond_result = True if self.map_cond is None else self.map_cond(new_array)
                        ready_to_yield = flux_cond_result and map_cond_result

                # Break
                if ready_to_yield:
                    break
                else:
                    self.cond_must_be_imposed = True
                    self.index_in_array += 1

            # Yield the sample
            # print("Yielding file", self.active_file, "index", self.index_in_array)
            if np.any(extra_info):
                # NOTE: The order of the models in the extra info dictionary is NOT the same as in MODELS!
                # This is because not all the fields are present for all the templates!
                # When using the extra info, make sure you know what you're looking at!
                extra_info_dict = {
                    info_key: np.asarray([self.data_dict["info"][info_key][model_key][self.index_in_array]
                                          for model_key in self.data_dict["info"][info_key].keys()])
                    for info_key in self.data_dict["info"].keys()}
                extra_info = dict_to_array(extra_info_dict)

                yield {"data": new_array.astype(np.float32), "label": labels,
                       "extra_info": extra_info.astype(np.float32)}
            else:
                yield {"data": new_array.astype(np.float32), "label": labels}

            self.index_in_array += 1
            self.cond_must_be_imposed = False

    def store_settings_dict(self, settings_dict):
        self.settings_dict = settings_dict


# U-Net
class PairGeneratorUNet(PairGeneratorOnTheFly):
    def __init__(self, estimate_templates=False, *args, **kwargs):
        super(PairGeneratorUNet, self).__init__(*args, **kwargs)
        self.est_templates = estimate_templates
        if len(self.NP_indices) > 0 and self.est_templates:
            raise NotImplementedError("Estimating templates is not supported for Non-Poissonian templates!")
        if np.any(np.asarray([*self.nside_PS.values()]) != self.nside):
            print("WARNING! Not all of the PS NSIDE parameters equal the selected NSIDE! "
                  "This means that fractional counts may occur due to the down/upsampling procedure, which should only "
                  "be used for debugging!")

    def get_next_pair(self, extra_info=False):
        """
        :return: next data - label pair (if EOF is reach: load from new file, if all files have been used: repeat)
        """
        while True:
            filenames = self.train_files_PS if self.is_training else self.test_files_PS
            train_str = "Training:" if self.is_training else "Validation:"

            # Initialise
            template_maps_counts = np.zeros((self.input_length, len(self.models)))
            A = np.zeros(len(self.models))
            var_weights = [None] * len(self.models)

            # Poissonian: draw the log amplitude and generate Poissonian sample
            for i_temp in self.P_indices:
                temp = self.models[i_temp]
                while template_maps_counts[:, i_temp].sum() == 0:  # require that there is at least a single count for each template!
                    # Draw template normalisation
                    A[i_temp] = random.uniform(self.prior_dict[temp][0], self.prior_dict[temp][1])
                    # Draw weighting of the variants randomly from a flat Dirichlet distribution
                    n_vars = len(self.model_vars[i_temp])
                    var_weights[i_temp] = np.random.dirichlet(np.ones(n_vars), size=1).flatten()
                    this_template_map = np.expand_dims(var_weights[i_temp], 0) @ np.asarray([self.template_maps[temp_var] for temp_var in self.model_vars[i_temp]])
                    if self.log_priors:
                        template_maps_counts[:, i_temp] = np.random.poisson((10.0 ** A[i_temp]) * this_template_map)
                    else:
                        template_maps_counts[:, i_temp] = np.random.poisson(A[i_temp] * this_template_map)

            # PS: load the sample
            for i_temp in self.NP_indices:
                assert len(self.model_vars[i_temp]) == 1, "Only 1 variant for each PS template is supported!"
                temp_var = self.model_vars[i_temp][0]

                # Get first file
                if len(self.active_file[temp_var]) == 0:
                    self.active_file[temp_var] = filenames[temp_var][self.file_no[temp_var]]
                    self.index_in_array[temp_var] = 0
                    print(train_str, "Opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                    try:
                        data_file = open(os.path.join(self.folder_maps_PS, temp_var, self.active_file[temp_var]), 'rb')
                        self.data_dict[temp_var] = pickle.load(data_file)
                        data_file.close()
                    except (FileNotFoundError, EOFError, IOError):
                        print(train_str, "Opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                        os._exit(2)

                # If end of a file
                elif self.index_in_array[temp_var] == self.data_dict[temp_var]["data"].shape[1]:
                    self.index_in_array[temp_var] = 0
                    self.file_no[temp_var] += 1
                    # If through all files: start at file 0
                    if self.file_no[temp_var] >= len(filenames[temp_var]):
                        self.file_no[temp_var] = 0
                        print(train_str, "All data from all files has been used for template", temp_var + ".")
                    self.active_file[temp_var] = filenames[temp_var][self.file_no[temp_var]]
                    print(train_str, "Opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                    try:
                        data_file = open(os.path.join(self.folder_maps_PS, temp_var, self.active_file[temp_var]), 'rb')
                        self.data_dict[temp_var] = pickle.load(data_file)
                        data_file.close()
                    except (FileNotFoundError, EOFError, IOError):
                        print(train_str, "Opening file", "..." + temp_var + "/" + self.active_file[temp_var], "...")
                        os._exit(2)

                # Get counts and A
                counts_raw = self.data_dict[temp_var]["data"][:, self.index_in_array[temp_var]]
                # if no up-/downsampling is needed: can be done quicker pre-computing the indices ..._get/_set
                if self.nside_PS[temp_var] == self.nside:
                    template_maps_counts[self.pixel_mapping_PS_set[temp_var], i_temp] = counts_raw[self.pixel_mapping_PS_get[temp_var]]
                else:
                    # Up/-downsample if needed (TAKES LONGER!)
                    counts_ring_full = masked_to_full(counts_raw, self.unmasked_pix_PS[temp_var], nside=self.nside_PS[temp_var])
                    counts_ring_full = hp.ud_grade(counts_ring_full, nside_out=self.nside, power=-2)
                    counts_ring_full_mask_applied = counts_ring_full * (1 - self.mask)
                    template_maps_counts[:, i_temp] = self.mask_fct(hp.reorder(counts_ring_full_mask_applied, r2n=True))

                var_weights[i_temp] = np.ones(1)
                A[i_temp] = self.data_dict[temp_var]["info"]["A"][self.index_in_array[temp_var]]

            # Calculate total input map (estimation is pixelwise, therefore, exposure correction does not need to be removed)
            input_map = template_maps_counts.sum(1)

            # Correct label: normalised templates or (noisy) template maps
            if self.est_templates:
                if self.log_priors:
                    labels = np.asarray([(10.0 ** A[i_temp]) * np.expand_dims(var_weights[i_temp], 0)
                                         @ np.asarray([self.template_maps[model_var] for model_var in self.model_vars[i_temp]]) for i_temp, temp in enumerate(self.models)]).T.squeeze(1)
                else:
                    labels = np.asarray([A[i_temp] * np.expand_dims(var_weights[i_temp], 0)
                                         @ np.asarray([self.template_maps[model_var] for model_var in self.model_vars[i_temp]]) for i_temp, temp in enumerate(self.models)]).T.squeeze(1)
            else:
                labels = template_maps_counts

            # Yield the sample
            if np.any(extra_info):
                yield {"data": input_map.astype(np.float32), "label": labels.astype(np.float32),
                       "var_fracs": flatten_var_fracs(var_weights).astype(np.float32),
                       "extra_info": A.astype(np.float32)}
            else:
                yield {"data": input_map.astype(np.float32), "label": labels.astype(np.float32),
                       "var_fracs": flatten_var_fracs(var_weights).astype(np.float32)}

            for key in self.index_in_array.keys():
                self.index_in_array[key] += 1


# # # # # # # # # # GCE Dataset classes # # # # # # # # # #
class Dataset(object):
    def __init__(self, n_params=1, batch_size=16, prefetch_batch_buffer=1, graph=None, nside=128,
                 trafo=lambda x: x, data_shape=None):
        """
        Base class for Tensorflow dataset objects
        :param generator: data generator based on which iterator will be built
        :param n_params: dimension of "label" (i.e. number of templates)
        :param batch_size: number of samples in a batch
        :param prefetch_batch_buffer: number of batches to prefetch
        :param graph: tensorflow graph associated to the dataset
        :param nside: nside of the data
        :param trafo: transformation that is applied to data (after removing exposure correction), lambda function
        :param data_shape: shape of data (photon counts), npix if entire sphere (unmask = True), otherwise determined by
                           length of get_pixels() output (when unmask = False)
        """
        self.n_params = n_params
        self.batch_size = batch_size
        self.prefetch_batch_buffer = prefetch_batch_buffer
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
        self.nside = nside
        self.data_trafo = trafo
        self.data_shape = data_shape

    def get_samples_base(self, n_samples, sess=None, n_params=None, UNet=False, return_var_fracs=False, return_gce_hist=False):
        """
        Sample generator method
        :param n_samples: number of samples to generate
        :param sess: tensorflow session (must be associated to this graph!)
        :param n_params: can be used to set a different number of expected flux fractions (useful when testing on a
                         different data set)
        :param UNet: if True: label shape is assumed to be n_batch x n_pix x n_templates
        :param return_var_fracs: if True: return fractions for template variants
        :param return_gce_hist: if True: return GCE histogram
        :return: data, labels, as produced by the tensorflow pipeline
        """
        if sess is None:
            sess = tf.compat.v1.Session(graph=self.graph)
        data_shape = self.data_shape or hp.nside2npix(self.nside)
        if not type(data_shape) is list:
            data_shape = [data_shape]
        out_data = np.zeros((0, *data_shape))
        if UNet:
            out_labels = np.zeros((0, data_shape[0], n_params or self.n_params))
        else:
            out_labels = np.zeros((0, n_params or self.n_params))
        if return_var_fracs:
            out_var_fracs = np.zeros((0, self.n_variants))
        if return_gce_hist:
            out_gce_hist = np.zeros((0, self.n_bins_hist))
        for i in range(int(np.ceil(n_samples / self.batch_size))):
            if return_var_fracs:
                if return_gce_hist:
                    next_data, next_labels, next_var_fracs, next_gce_hist = \
                        sess.run([self.next_element.data, self.next_element.label, self.next_element.var_fracs, self.next_element.gce_hist])
                else:
                    next_data, next_labels, next_var_fracs = \
                        sess.run([self.next_element.data, self.next_element.label, self.next_element.var_fracs])
            else:
                if return_gce_hist:
                    next_data, next_labels, next_gce_hist = sess.run([self.next_element.data, self.next_element.label, self.next_element.gce_hist])
                else:
                    next_data, next_labels = sess.run([self.next_element.data, self.next_element.label])
            out_data = np.concatenate([out_data, next_data], 0)
            out_labels = np.concatenate([out_labels, next_labels], 0)
            if return_var_fracs:
                out_var_fracs = np.concatenate([out_var_fracs, next_var_fracs], 0)
            if return_gce_hist:
                out_gce_hist = np.concatenate([out_gce_hist, next_gce_hist], 0)

        out_data = out_data[:n_samples, :]
        out_labels = out_labels[:n_samples, :]
        if return_var_fracs:
            out_var_fracs = out_var_fracs[:n_samples, :]

        if return_gce_hist:
            out_gce_hist = out_gce_hist[:n_samples, :]

        out_dict = dict()
        out_dict["data"], out_dict["label"] = out_data, out_labels
        if return_var_fracs:
            out_dict["var_fracs"] = out_var_fracs
        if return_gce_hist:
            out_dict["gce_hist"] = out_gce_hist
        return out_dict

    def get_fermi_counts(self, fermi_path, fermi_name="fermidata_counts", sess=None, rescale=None, indices=None,
                         outer_mask=None, remove_plane=2.0, fill_val=0.0, mask_3FGL=False, only_hemisphere=None,
                         nside=128, remove_exp=True):
        """Get the counts from the Fermi map and do the same pre-processing as for the training data.
           NOTE: rescale and indices must be in NESTED format!"""
        if sess is None:
            sess = tf.compat.v1.Session(graph=self.graph)
        fermi_data = np.load(os.path.join(fermi_path, fermi_name + ".npy"))

        # Mask Galactic plane up to "remove_plane" degrees
        total_mask_neg = cm.make_mask_total(band_mask=remove_plane > 0, band_mask_range=remove_plane,
                                            mask_ring=outer_mask is not None, inner=0, outer=outer_mask)
        if mask_3FGL:
            total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_path, "3FGL_mask")).astype(bool)).astype(bool)

        if only_hemisphere is not None:
            total_mask_pos_N, total_mask_pos_S = split_into_N_and_S(1 - total_mask_neg, nside=hp.npix2nside(len(fermi_data)), filename=None)
            total_mask_neg = (1 - total_mask_pos_S) if only_hemisphere == "S" else (1 - total_mask_pos_N)
            total_mask_neg = total_mask_neg.astype(bool)

        fermi_data[total_mask_neg] = fill_val

        # Ring -> nested
        fermi_data = hp.reorder(fermi_data, r2n=True)

        # Downscale
        if nside != 128:
            fermi_data = hp.ud_grade(fermi_data, nside_out=nside, order_in="NEST", order_out="NEST")

        # Reduce to ROI
        if indices is not None:
            fermi_data = fermi_data[indices]

        # Remove exposure correction
        if rescale is not None and remove_exp:
            fermi_data = remove_exposure_correction(fermi_data, rescale=rescale)  # rescale (rescale: nest & ROI)

        # Apply transformation
        fermi_data = self.data_trafo(fermi_data)

        # Get numpy array
        if tf.is_tensor(fermi_data):
            fermi_data = np.float64(sess.run(fermi_data))
        elif type(fermi_data) is np.ndarray:
            fermi_data = np.float64(fermi_data)
        else:
            raise NotImplementedError
        return fermi_data


# CNN
class DatasetCNN(Dataset):
    def __init__(self, generator, n_params=1, n_variants=1, batch_size=16, prefetch_batch_buffer=1, graph=None, nside=128,
                 trafo=lambda x: x, data_shape=None, var_fracs=True, gce_hist=False, n_bins_hist=0):
        super(DatasetCNN, self).__init__(n_params, batch_size, prefetch_batch_buffer, graph, nside, trafo, data_shape)
        self.var_fracs = var_fracs
        self.gce_hist = gce_hist
        self.n_bins_hist = n_bins_hist
        self.n_variants = n_variants
        self.next_element = self.build_iterator(generator)
        self.next_element_with_info = self.build_iterator(generator, extra_info=True)

    def build_iterator(self, pair_gen: PairGenerator, extra_info=False):
        """
        Build the iterator for tensorflow using the Dataset class
        :param pair_gen: PairGenerator object
        :return: The iterator returns a GCEData object (dictionary containing "data" and "label")
        """
        output_type_dict = {"data": tf.float32, "label": tf.float32}
        if extra_info:
            output_type_dict["extra_info"] = tf.float32
        if self.var_fracs:
            output_type_dict["var_fracs"] = tf.float32
        if self.gce_hist:
            output_type_dict["gce_hist"] = tf.float32

        with self.graph.as_default():
            if extra_info:
                dataset = tf.data.Dataset.from_generator(lambda: pair_gen.get_next_pair(extra_info=True), output_types=output_type_dict)
            else:
                dataset = tf.data.Dataset.from_generator(pair_gen.get_next_pair, output_types=output_type_dict)
            dataset = dataset.map(lambda x: self.set_size_and_map(x, n_params=self.n_params),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.prefetch_batch_buffer)
            iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
            element = iter.get_next()

        var_fracs = element["var_fracs"] if self.var_fracs else None
        extra_info = element["extra_info"] if extra_info else None
        gce_hist = element["gce_hist"] if self.gce_hist else None

        return GCEData(element["data"], element["label"], var_fracs, extra_info, gce_hist)


    def get_samples(self, n_samples, sess=None, n_params=None):
        return_var_fracs = True if self.var_fracs else False
        return_gce_hist = True if self.gce_hist else False
        return self.get_samples_base(n_samples, sess, n_params, UNet=False, return_var_fracs=return_var_fracs, return_gce_hist=return_gce_hist)

    def set_size_and_map(self, pair_element, n_params=1):
        """
        NOTE: exposure correction has already been removed in PairGenerator.get_next_pair()
        Set the size of the tensors and apply a transformation if requested
        :param pair_element: dataset from generator
        :param n_params: dimension of "label" (i.e. number of templates)
        :return: rescaled, mapped pair_element (with sizes set)
        """
        # Set shapes
        data_shape = self.data_shape or hp.nside2npix(self.nside)
        if not type(data_shape) is list:
            data_shape = [data_shape]

        pair_element["data"].set_shape(data_shape)
        pair_element["label"].set_shape([n_params])
        if self.var_fracs:
            pair_element["var_fracs"].set_shape(self.n_variants)
        if self.gce_hist:
            pair_element["gce_hist"].set_shape(self.n_bins_hist)

        # Do pixelwise data transformation
        pair_element["data"] = self.data_trafo(pair_element["data"])

        return pair_element  # NOTE: do NOT do the data rescaling to relative counts here!
        # This will be done inside the NN in order to save the total number of counts and append it

    def get_additional_test_data(self, models, data_file, settings_file, fermi_path, nside=128, sess=None, rescale=None,
                                 indices=None, outer_mask=None, remove_exp=True,
                                 remove_plane=2.0, fill_val=0.0, mask_3FGL=False, input_is_ring=True):
        """Get additional test data (e.g. simulated Fermi maps corresponding to the best fit parameters), and do the
        same pre-processing as for the training data. NOTE: rescale and indices must be in NESTED format!"""
        if sess is None:
            sess = tf.compat.v1.Session(graph=self.graph)

        data = pickle.load(open(data_file, 'rb'))
        maps = data["data"].T  # saved format: Npix x Nsamples -> transpose
        labels_new = np.asarray([data["flux_fraction"][key] for key in models]).T
        settings = pickle.load(open(settings_file, 'rb'))

        # Get entire map
        maps_full = masked_to_full(maps, settings["unmasked_pix"], nside=nside)

        # Mask Galactic plane up to "remove_plane" degrees
        total_mask_neg = cm.make_mask_total(band_mask=remove_plane > 0, band_mask_range=remove_plane,
                                            mask_ring=outer_mask is not None, inner=0, outer=outer_mask)
        if mask_3FGL:
            total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_path, "3FGL_mask")).astype(bool)).astype(bool)
        maps_full[:, total_mask_neg] = fill_val

        # Ring -> nested
        maps_full = hp.reorder(maps_full, r2n=True) if input_is_ring else maps_full

        # Reduce to ROI
        maps_new = maps_full[:, indices] if indices is not None else maps_full

        # Remove exposure correction
        if rescale is not None and remove_exp:
            maps_new = remove_exposure_correction(maps_new, rescale=rescale)  # rescale (rescale: nest & ROI)

        # Apply transformation
        maps_new = self.data_trafo(maps_new)

        # Get numpy array
        if tf.is_tensor(maps_new):
            maps_new = np.float64(sess.run(maps_new))
        elif type(maps_new) is np.ndarray:
            maps_new = np.float64(maps_new)
        else:
            raise NotImplementedError
        return maps_new, labels_new


# U-Net
class DatasetUNet(Dataset):
    def __init__(self, generator, n_variants=1, n_params=1, batch_size=16, prefetch_batch_buffer=1, graph=None,
                 nside=128, trafo=lambda x: x, data_shape=None):
        super(DatasetUNet, self).__init__(n_params, batch_size, prefetch_batch_buffer, graph, nside, trafo, data_shape)
        self.n_variants = n_variants
        self.next_element = self.build_iterator(generator)
        self.next_element_with_info = self.build_iterator(generator, extra_info=True)

    def build_iterator(self, pair_gen: PairGenerator, extra_info=False):
        """
        Build the iterator for tensorflow using the Dataset class
        :param pair_gen: PairGenerator object
        :return: The iterator returns a GCEData object (dictionary containing "data" and "label")
        """
        with self.graph.as_default():
            if extra_info:
                dataset = tf.data.Dataset.from_generator(lambda: pair_gen.get_next_pair(extra_info=True),
                                                         output_types={"data": tf.float32,
                                                                       "label": tf.float32,
                                                                       "var_fracs": tf.float32,
                                                                       "extra_info": tf.float32})
            else:
                dataset = tf.data.Dataset.from_generator(pair_gen.get_next_pair,
                                                         output_types={"data": tf.float32,
                                                                       "label": tf.float32,
                                                                       "var_fracs": tf.float32})
            dataset = dataset.map(lambda x: self.set_size_and_map(x, n_params=self.n_params),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.prefetch_batch_buffer)
            iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
            element = iter.get_next()
        if extra_info:
            return GCEData(element["data"], element["label"], element["var_fracs"], element["extra_info"])
        else:
            return GCEData(element["data"], element["label"], element["var_fracs"])



    def get_samples(self, n_samples, sess=None, n_params=None):
        return self.get_samples_base(n_samples, sess, n_params, UNet=True, return_var_fracs=False)

    def set_size_and_map(self, pair_element, n_params=1):
        """
        Set the size of the tensors and apply a transformation if requested
        :param pair_element: dataset from generator
        :param n_params: first dimension of "label" (i.e. number of templates)
        :return: mapped pair_element (with sizes set)
        """
        # Set shapes
        data_shape = self.data_shape or hp.nside2npix(self.nside)
        if not type(data_shape) is list:
            data_shape = [data_shape]

        pair_element["data"].set_shape(data_shape)
        pair_element["label"].set_shape([data_shape[0], n_params])
        pair_element["var_fracs"].set_shape(self.n_variants)

        # Do pixelwise data transformation
        pair_element["data"] = self.data_trafo(pair_element["data"])

        return pair_element  # NOTE: do NOT do the data rescaling to relative counts here!
        # This will be done inside the NN
