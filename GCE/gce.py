"""
This file defines the main class of the neural network-based GCE analysis.
"""
import os
import numpy as np
import healpy as hp
import time
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import pprint
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from .tf_ops import get_gpu_names
from .utils import import_from, DotDict
from .data_generation import generate_template_maps
from .data_combination import combine_template_maps
from .data_utils import build_index_dict, masked_to_full, make_mask_total, get_template
from .parameter_utils import get_subdict, load_params_from_pickle
from .nn.pipeline import build_pipeline
from .nn.Models.deepsphere_cnn import DeepsphereCNN
from .nn import losses
from .plots import plot_flux_fractions, plot_histograms, plot_maps


# TODO:
# * for dN/dF labels: take CDF of true underlying skew normal dist. instead of histograms (although: how to deal with multiple populations?)
# * add checks for continuous dN/dF


class Analysis:
    def __init__(self):
        """
        Initialise Analysis object.
        """
        self._gpu_names = get_gpu_names()
        self.gpu_found = len(self._gpu_names) > 0
        self.p = DotDict()
        self.template_dict = {}
        self.generators = {}
        self.datasets = {}
        self.inds = {}
        self._nn_spec = None
        self.nn = None
        self._trainable_weights_dict = {}
        self.set_plot_defaults()
        self._strategy = tf.distribute.MirroredStrategy()
        self._gce_module_folder = os.path.dirname(os.path.abspath(__file__))
        print('Number of devices: {}'.format(self._strategy.num_replicas_in_sync))

    def load_params(self, param_src, int_flag=0, overwrite=False):
        """
        Load the parameters.
        :param param_src: can be a python parameter file or a folder containing a parameter pickle file
        :param int_flag: an integer flag that will be passed to 'get_params()', only used if params is a python file
        :param overwrite: if True: overwrite settings already contained in self.p
        """
        print("Loading parameters from", "'" + param_src + "'...")
        # Load from python parameter file
        if param_src.endswith(".py"):
            par_fun = import_from(param_src, "get_params")
            loaded_dict = par_fun(int_flag)
            self._load_params_intern(loaded_dict, overwrite)
        # Load from pickle file
        else:
            self._load_params_from_pickle(param_src, overwrite=overwrite)
        self._process_params()
        self._check_params()
        required_keys = ("gen", "data", "mod")
        self._check_keys_exist(required_keys)
        self._store_template_dict()

    def _load_params_from_pickle(self, folder, overwrite=False):
        """
        Load parameters from pickle file.
        Note: the parameters can come from template maps, combined maps, etc.
        :param folder: folder name
        :param overwrite: if True: overwrite settings already contained in self.p
        """
        loaded_dict = load_params_from_pickle(folder, return_normal_dict=False)
        self._load_params_intern(loaded_dict, overwrite)

    def _load_params_intern(self, loaded_dict, overwrite):
        """
        Internal method to update parameters
        :param loaded_dict: loaded dictionary
        :param overwrite: if True: overwrite settings already contained in self.p
        """
        new_keys = np.setdiff1d(list(loaded_dict.keys()), list(self.p.keys()))  # keys in loaded but not in self.p
        same_keys = np.setdiff1d(list(loaded_dict.keys()), new_keys)  # loaded keys that exist already

        if overwrite:
            if len(same_keys) > 0:
                print("Values of keys", same_keys, "are overwritten!")
                print("Additionally loading:", new_keys, ".")
            else:
                print("Loading parameters:", new_keys)
            for k in loaded_dict.keys():
                self.p[k] = loaded_dict[k]
        else:
            if len(new_keys) == 0:
                print("No new parameters found and 'overwrite=False'! No parameters were loaded.")
            else:
                print("Loading parameters:", new_keys)
                for k in new_keys:
                    self.p[k] = loaded_dict[k]

    def _check_keys_exist(self, required_keys):
        """
        Check that the parameter dictionary contains all the necessary keys.
        :param required_keys: iterable containing the keys
        """
        assert np.all([k in self.p.keys() for k in required_keys]), \
            "Parameters are missing! Required keys: {:}, found keys: {:}".format(required_keys, self.p.keys())

    def _check_params(self):
        """
        Check that parameters are correct and consistent.
        """
        if "data" in self.p.keys():
            assert self.p.data["mask_type"] in [None, "None", "3FGL", "4FGL"], \
                "Mask type not recognised!"
            if isinstance(self.p.data["exposure"], str) and not "npy" in self.p.data["exposure"]:
                assert self.p.data["exposure"] in ["Fermi", "Fermi_mean"], \
                    "Exposure not recognised!"
            assert hp.isnsideok(self.p.data["nside"]), "nside is not OK!"
        if "tt" in self.p.keys():
            assert np.all(["PS" in t for t in self.p.tt["add_two_temps_PS"]]), \
                "Not all the templates in 'add_two_temps_PS' seem to be PS templates!"
        if "mod" in self.p.keys():
            assert len(self.p.mod["models_P"]) == len(self.p.mod["model_names_P"]), \
                "Lengths of Poissonian models and model names do not match!"
            assert len(self.p.mod["models_PS"]) == len(self.p.mod["model_names_PS"]), \
                "Lengths of point-source models and model names do not match!"
            assert np.all([mod.endswith("_PS") for mod in self.p.mod["models_PS"]]), \
                "All PS models must end with '_PS'!"
            assert np.all(["_PS" not in mod for mod in self.p.mod["models_P"]]), \
                "Poissonian models should not contain '_PS'!"
        if "tt" in self.p.keys() and "mod" in self.p.keys():
            assert np.all([t in self.p.tt["priors"] for t in self.p.mod["models_P"]]), \
                "Missing prior specifications for Poissonian model(s)!"
            assert np.all([t in self.p.tt["priors"] for t in self.p.mod["models_PS"]]), \
                "Missing prior specifications for point-source model(s)!"
        if "data" in self.p.keys() and "comb" in self.p.keys():
            if self.p.comb["combine_without_PSF"]:
                assert self.p.data["psf"], "Option clash: 'combine_without_PSF' requires setting 'PSF = True'."
        if "data" in self.p.keys() and "nn" in self.p.keys():
            assert self.p.data["nside"] == self.p.nn.arch["nsides"][0],\
                "nside = {:}, but nsides[0] = {:}!".format(self.p.data["nside"], self.p.nn.arch["nsides"][0])
        if "comb" in self.p.keys() and "nn" in self.p.keys() and self.p.nn.hist["return_hist"]:
            assert np.all([temp in self.p.comb["hist_templates"] for temp in self.p.nn.hist["hist_templates"]]), \
                "self.p.nn.hist['hist_templates'] must be a subset of self.p.comb['hist_templates']!"
            assert self.p.comb["do_" + self.p.nn.hist["which_histogram"]], \
                "Histogram '{:}' was not saved for combined maps!".format(self.p.nn.hist["which_histogram"])
        if "nn" in self.p.keys():
            assert len(self.p.nn.arch["nsides"]) - 1 == len(self.p.nn.arch["F"]) == len(self.p.nn.arch["K"]) == \
                   len(self.p.nn.arch["is_resnet"]), "Specification of NN architecture is not correct!"
            assert len(self.p.nn.arch["batch_norm"]) == len(self.p.nn.arch["F"]) + len(self.p.nn.arch["M"]), \
                "Normalization must be set for all convolutional and fully-connected layers. Expected length: {:}," \
                "found length: {:}.".format(len(self.p.nn.arch["F"]) + len(self.p.nn.arch["M"]),
                                            len(self.p.nn.arch["batch_norm"]))
            assert not (self.p.nn.ff["alea_var"] and self.p.nn.ff["alea_covar"]), \
                "Choose either 'alea_var' or 'alea_covar', not both!"
            assert not self.p.nn.hist["calculate_residual"] or \
                   (self.p.nn.ff["return_ff"] and self.p.nn.hist["return_hist"]), \
                    "The option 'calculate_residual' requires 'return_ff' == 'return_hist' == True!"
        if "train" in self.p.keys():
            assert self.p.train['hist_pinball_smoothing'] > 0, "Pinball smoothing must be non-negative!"
        if "nn" in self.p.keys() and "train" in self.p.keys():
            assert not (self.p.nn.ff["alea_var"] and self.p.nn.ff["alea_covar"]) or self.p.train["ff_loss"] == "l2", \
                "Flux fraction uncertainty estimation requires self.p.train['ff_loss'] = 'l2'!"
        if "plot" in self.p.keys() and "mod" in self.p.keys():
            assert len(self.p.plot["colors_P"]) >= len(self.p.mod["models_P"]) and \
                   len(self.p.plot["colors_PS"]) >= len(self.p.mod["models_PS"]), "Not enough colors provided!"

    def _process_params(self):
        """
        Add some convenient keys to the parameter dictionary that can be derived from the user-specified parameters.
        """
        nside = self.p.data["nside"]

        # Root folders for checkpoints, summaries, parameters, and figures
        if "gen" in self.p.keys():
            self.p.gen["checkpoints_root"] = os.path.join(self.p.gen["models_root"], "checkpoints")
            self.p.gen["summaries_root"] = os.path.join(self.p.gen["models_root"], "summaries")
            self.p.gen["params_root"] = os.path.join(self.p.gen["models_root"], "parameters")
            self.p.gen["figures_root"] = os.path.join(self.p.gen["models_root"], "figures")

        # Data folders
        if "gen" in self.p.keys():
            self.p.gen["fermi_folder"] = os.path.join(self.p.gen["fermi_root"],
                                                      "fermi_data_" + str(nside))
        if "gen" in self.p.keys() and "tt" in self.p.keys():
            self.p.gen["template_maps_folder"] = os.path.join(self.p.gen["template_maps_root"],
                                                              self.p.tt["data_name"] + "_" + str(nside))
        if "gen" in self.p.keys() and "comb" in self.p.keys():

            self.p.gen["combined_maps_folder"] = os.path.join(self.p.gen["combined_maps_root"],
                                                              self.p.comb["data_name"] + "_" + str(nside))

        # Also store npix
        if "data" in self.p.keys():
            self.p.data["npix"] = hp.nside2npix(self.p.data["nside"])

        # Combined P and PS models, names, and lengths
        if "mod" in self.p.keys():
            self.p.mod["models"] = self.p.mod["models_P"] + self.p.mod["models_PS"]
            self.p.mod["model_names"] = self.p.mod["model_names_P"] + self.p.mod["model_names_PS"]
            self.p.mod["n_models_P"] = len(self.p.mod["models_P"])
            self.p.mod["n_models_PS"] = len(self.p.mod["models_PS"])
            self.p.mod["n_models"] = self.p.mod["n_models_P"] + self.p.mod["n_models_PS"]

        # Folders for checkpoints, summaries, and figures
        if "comb" in self.p.keys() and "nn" in self.p.keys():
            self.p.nn.hist["nn_hist_bins"] = getattr(self.p.comb, "bins_" + self.p.nn.hist["which_histogram"])
            data_str = self.p.comb["data_name"] + "_" + str(nside)
            run_str = self.p.nn["run_name"]
            self.p.nn["checkpoints_folder"] = os.path.join(self.p.gen["checkpoints_root"], data_str, run_str)
            self.p.nn["summaries_folder"] = os.path.join(self.p.gen["summaries_root"], data_str, run_str)
            for k in ["checkpoints_folder", "summaries_folder"]:
                for sub in ["ff", "hist"]:
                    self.p.nn[k + "_" + sub] = os.path.join(self.p.nn[k], sub)
            self.p.nn["params_folder"] = os.path.join(self.p.gen["params_root"], data_str, run_str)
            self.p.nn["figures_folder"] = os.path.join(self.p.gen["figures_root"], data_str, run_str)

        # Label shapes
        if "mod" in self.p.keys() and "nn" in self.p.keys():
            if self.p.nn.hist["return_hist"]:
                n_hist_templates = len(self.p.nn.hist["hist_templates"])
                self.p.nn.hist["n_bins"] = len(self.p.nn.hist["nn_hist_bins"]) - 1
                len(self.p.nn.hist["hist_templates"])
                n_bins = self.p.nn.hist["n_bins"]
                self.p.nn["label_shape"] = [[self.p.mod["n_models"], len(self.p.data["log_ebins"]) - 1],
                                            [n_bins, n_hist_templates]]
                # fluxes, SCD histograms
            else:
                self.p.nn["label_shape"] = [[self.p.mod["n_models"], len(self.p.data["log_ebins"]) - 1]]  # fluxes

        # NN output keys
        if "nn" in self.p.keys():
            output_keys_ff = []
            if self.p.nn.ff["return_ff"]:
                output_keys_ff.append("ff_mean")
                if self.p.nn.ff["alea_var"]:
                    output_keys_ff.append("ff_logvar")
                if self.p.nn.ff["alea_covar"]:
                    output_keys_ff.append("ff_covar")
            output_keys_hist = []
            if self.p.nn.hist["return_hist"]:
                output_keys_hist.append("hist")
            self.p.nn.ff.output_keys = output_keys_ff
            self.p.nn.hist.output_keys = output_keys_hist
            self.p.nn.output_keys = self.p.nn.ff.output_keys + self.p.nn.hist.output_keys

        # Does the NN require tau as an input?
        if "nn" in self.p.keys() and "train" in self.p.keys():
            if self.p.nn.hist["return_hist"] and "EMPL" in self.p.train["hist_loss"].upper():
                self.p.nn.requires_tau = True
            else:
                self.p.nn.requires_tau = False

        # If a condition is given: run 'eval' on string (user needs to be careful!)
        if "nn" in self.p.keys():
            if self.p.nn.cond["cond_on_training_data_str"] is not None:
                self.p.nn.cond["cond_on_training_data"] = eval(self.p.nn.cond["cond_on_training_data_str"])
            else:
                self.p.nn.cond["cond_on_training_data"] = None
            if self.p.nn.cond["cond_on_training_labels_str"] is not None:
                self.p.nn.cond["cond_on_training_labels"] = eval(self.p.nn.cond["cond_on_training_labels_str"])
            else:
                self.p.nn.cond["cond_on_training_labels"] = None

        # Process histogram bins
        if "nn" in self.p.keys():
            if self.p.nn.hist["return_hist"]:
                bin_edges = self.p.nn.hist["nn_hist_bins"]
                bin_edges_mapped = np.log10(bin_edges) if self.p.nn.hist["log_spaced_bins"] else bin_edges
                if bin_edges[0] == -np.infty:  # if left-most bin contains everything fainter than bin range
                    bin_edges_mapped[0] = 2 * bin_edges_mapped[1] - bin_edges_mapped[2]
                if bin_edges[-1] == np.infty:  # if right-most bin contains everything brighter than bin range
                    bin_edges_mapped[-1] = 2 * bin_edges_mapped[-2] - bin_edges_mapped[-3]
                bin_centers_mapped = (bin_edges_mapped[1:] + bin_edges_mapped[:-1]) / 2.0
                bin_centers = 10 ** bin_centers_mapped if self.p.nn.hist["log_spaced_bins"] else bin_centers_mapped
                bin_edges_mapped_inv = 10 ** bin_edges_mapped if self.p.nn.hist["log_spaced_bins"] else bin_edges_mapped

                self.p.nn.hist["nn_hist_centers"] = bin_centers
                self.p.nn.hist["nn_hist_widths"] = np.diff(bin_edges_mapped_inv)
                self.p.nn.hist["nn_hist_widths_l"] = bin_centers - bin_edges_mapped_inv[:-1]
                self.p.nn.hist["nn_hist_widths_r"] = bin_edges_mapped_inv[1:] - bin_centers

        # Store histogram template indices and names
        if "nn" in self.p.keys() and "mod" in self.p.keys():
            if self.p.nn.hist["return_hist"]:
                assert np.all([t in self.p.mod["models"] for t in self.p.nn.hist["hist_templates"]]), \
                    "Not all hist_templates are contained in self.p.mod['models']!"
                self.p.nn.hist["hist_template_inds"] = np.asarray(([np.argwhere([mod == t
                                                                                 for mod in self.p.mod["models"]])
                                                                    for t in self.p.nn.hist["hist_templates"]])
                                                                  ).flatten()
                self.p.nn.hist["hist_template_names"] = [self.p.mod["model_names"][i]
                                                         for i in self.p.nn.hist["hist_template_inds"]]

        # Plotting
        if "plot" in self.p.keys():
            self.p.plot["colors"] = self.p.plot["colors_P"] + self.p.plot["colors_PS"]

        # Delete unused prior specifications
        if "tt" in self.p.keys() and "mod" in self.p.keys():
            if "priors" in self.p.tt.keys():
                self.p.tt.priors = {temp: self.p.tt.priors[temp] for temp in self.p.mod["models"]}

    def print_params(self, subdict=None, keys_only=False):
        """
        Print currently stored parameter dictionary
        :param subdict: str or None. If str, name of subdictionary to print
        :param keys_only: only print keys (non-recursively)?
        """
        if subdict is None:
            if keys_only:
                pprint.pprint(self.p.keys())
            else:
                pprint.pprint(self.p)
        else:
            if subdict in self.p.keys():
                if keys_only:
                    pprint.pprint(self.p[subdict].keys())
                else:
                    pprint.pprint(self.p[subdict])
            else:
                KeyError("Dictionary does not contain key '{:}'".format(subdict))

    def _store_template_dict(self):
        """
        Store specified templates in a dictionary, together with exposure and mask.
        """
        temp_dict = DotDict()

        # Set shortcuts
        fermi_folder = self.p.gen["fermi_folder"]
        inner_band = self.p.data["inner_band"]
        outer_rad = self.p.data["outer_rad"]
        lon_min = self.p.data["lon_min"]
        lon_max = self.p.data["lon_max"]
        lat_min = self.p.data["lat_min"]
        lat_max = self.p.data["lat_max"]
        nside = self.p.data["nside"]

        # Set up the mask for the ROI
        total_mask_neg = make_mask_total(band_mask=True, band_mask_range=inner_band, mask_ring=True, inner=0,
                                         outer=outer_rad, nside=nside, l_deg_min=lon_min, l_deg_max=lon_max,
                                         b_deg_min=lat_min, b_deg_max=lat_max, b_mask=True, l_mask=True)

        if self.p.data["mask_type"] == "3FGL":
            total_mask_neg = (1 - (1 - total_mask_neg)
                              * (1 - get_template(fermi_folder, "3FGL_mask"))).astype(bool)
        elif self.p.data["mask_type"] == "4FGL":
            total_mask_neg = (1 - (1 - total_mask_neg)
                              * (1 - get_template(fermi_folder, "4FGL_mask"))).astype(bool)
        elif self.p.data["mask_type"] not in [None, "None"]:
            warnings.warn("Warning! The mask type self.p.data['mask_type'] = '{:}' is not recognized and will be "
                          "ignored. Choose '3FGL', '4FGL', or None.".format(self.p.data["mask_type"]))

        # Set up a slightly larger mask that allows point-source counts to be smeared into the actual ROI by the PSF
        leakage_delta = self.p.data["leakage_delta"] if self.p.data["psf"] else 0
        if not self.p.data["psf"] and self.p.data["leakage_delta"] > 0:
            warnings.warn("Leakage delta > 0 will be ignored because of self.p.data['psf'] = False!")

        if leakage_delta > 0:
            total_mask_neg_safety = make_mask_total(band_mask=True, band_mask_range=max(0, inner_band - leakage_delta),
                                                    mask_ring=True, inner=0, outer=outer_rad + leakage_delta,
                                                    nside=nside,
                                                    l_deg_min=lon_min - leakage_delta, l_deg_max=lon_max + leakage_delta,
                                                    b_deg_min=lat_min - leakage_delta, b_deg_max=lat_max + leakage_delta,
                                                    b_mask=True, l_mask=True)
        else:
            total_mask_neg_safety = total_mask_neg.copy()

        # RING -> NEST
        total_mask_neg = hp.reorder(total_mask_neg, r2n=True)
        total_mask_neg_safety = hp.reorder(total_mask_neg_safety, r2n=True)
        indices_roi = np.argwhere(~total_mask_neg).flatten()
        indices_safety = np.argwhere(~total_mask_neg_safety).flatten()

        # Store mask and corresponding indices
        temp_dict["mask_ROI_full"] = total_mask_neg
        temp_dict["mask_safety_full"] = total_mask_neg_safety
        temp_dict["indices_roi"] = indices_roi
        temp_dict["indices_safety"] = indices_safety

        # Get exposure map
        if "npy" in self.p.data["exposure"]:
            fermi_exp = hp.reorder(np.load(self.p.data["exposure"]), r2n=True)
            # if it has an energy dimension: swap axes
            if fermi_exp.ndim == 2:
                fermi_exp = np.swapaxes(fermi_exp, 0, 1)
        else:
            fermi_exp = hp.reorder(get_template(fermi_folder, "exp"), r2n=True)
        fermi_exp_compressed = fermi_exp[indices_roi]
        fermi_mean_exp_roi = fermi_exp_compressed.mean()

        if "npy" in self.p.data["exposure"]:
            exp = fermi_exp.copy()  # NOTE: diffuse model has not yet been exposure corrected!
        elif self.p.data["exposure"] == "Fermi":
            exp = fermi_exp.copy()
        elif self.p.data["exposure"] == "Fermi_mean":
            exp = np.ones_like(fermi_exp) * fermi_mean_exp_roi
        elif isinstance(self.p.data["exposure"], (int, float)):
            exp = np.ones_like(fermi_exp) * self.p.data["exposure"]
        else:
            raise NotImplementedError

        exp_compressed = exp[indices_roi]
        mean_exp_roi = exp_compressed.mean()

        # Store exposure for Fermi map (always needed because templates are stored exposure corrected!)
        temp_dict["fermi_exp"] = fermi_exp
        temp_dict["fermi_exp_compressed"] = fermi_exp_compressed
        temp_dict["fermi_mean_exp_roi"] = fermi_mean_exp_roi
        temp_dict["fermi_rescale"] = fermi_exp / fermi_mean_exp_roi
        temp_dict["fermi_rescale_compressed"] = fermi_exp_compressed / fermi_mean_exp_roi

        # Store exposure for data considered here
        temp_dict["exp"] = exp
        temp_dict["exp_compressed"] = exp_compressed
        temp_dict["mean_exp_roi"] = mean_exp_roi
        temp_dict["rescale"] = exp / mean_exp_roi
        temp_dict["rescale_compressed"] = exp_compressed / mean_exp_roi

        # Iterate over the templates
        t_p = self.p.mod["models_P"]
        t_ps = self.p.mod["models_PS"]

        temp_dict["T_counts"] = {}
        temp_dict["T_flux"] = {}
        temp_dict["counts_to_flux_ratio_roi"] = {}

        # Now, store the templates
        for temp in t_p + t_ps:
            temp_p_name = temp[:-3] if "_PS" in temp else temp
            smooth = "_PS" not in temp
            t = hp.reorder(get_template(fermi_folder, temp_p_name, smooth=smooth), r2n=True)
            if t.ndim == 2:
                t = np.swapaxes(t, 0, 1)

            # if data has Fermi exposure
            if self.p.data["exposure"] == "Fermi" or "npy" in self.p.data["exposure"]:
                t_counts = t  # exposure-corrected template
                t_flux = t_counts / temp_dict["fermi_rescale"]  # after removing the exposure correction

            # if exposure is constant: Fermi exposure correction is removed for flux and counts template
            else:
                t_counts = t_flux = t / temp_dict["fermi_rescale"]

            counts_to_flux_ratio_roi = t_counts[indices_roi].sum() / t_flux[indices_roi].sum()

            temp_dict["T_counts"][temp] = t_counts
            temp_dict["T_flux"][temp] = t_flux
            temp_dict["counts_to_flux_ratio_roi"][temp] = counts_to_flux_ratio_roi

        self.template_dict = temp_dict

    def generate_template_maps(self, ray_settings, n_example_plots=5, job_id=0):
        """
        Generate simulated template maps that can later be combined for training, validation, and testing.
        :param ray_settings: settings passed to ray when calling ray.init()
        :param n_example_plots: number of maps to plot and save for each template (as a quick check)
        :param job_id: if running several jobs for the data generation: ID of the current job
        """
        required_keys = ("gen", "data", "mod", "tt")
        self._check_keys_exist(required_keys)

        os.makedirs(self.p.gen["data_root"], exist_ok=True)
        os.makedirs(self.p.gen["template_maps_root"], exist_ok=True)
        os.makedirs(self.p.gen["fermi_root"], exist_ok=True)

        # Check if data exists already
        if os.path.isdir(self.p.gen["template_maps_folder"]):
            template_folder_content = os.listdir(self.p.gen["template_maps_folder"])
            if np.any(["params" in filename and filename.endswith(".pickle") for filename in template_folder_content]):
                print("Template maps exist already!")
                return

        # Generate template maps
        generate_template_maps(self.p, self.template_dict, ray_settings, n_example_plots, job_id)

        # Save parameters
        param_subdict = get_subdict(self.p, keys=required_keys, return_normal_dict=True)
        datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
        with open(os.path.join(self.p.gen["template_maps_root"],
                               self.p.tt["data_name"] + "_" + str(self.p.data["nside"]),
                  "params_" + datetime + ".pickle"), "wb") as params_file:
            pickle.dump(param_subdict, params_file)

    def combine_template_maps(self, job_id=0, train_range=None, val_range=None, test_range=None,
                              save_filenames=True, do_combine=True, verbose=False):
        """
        Combine simulated template maps to training, validation, and testing maps.
        This is done in 2 steps:
        1. The filenames of the template maps that will be combined are stored.
        2. The template maps are combined and saved.
        :param job_id: if running several jobs for the data generation: ID of the current job
        :param train_range: [first index, last index] for training data handled by this job
        :param val_range:  [first index, last index] for validation data handled by this job
        :param test_range:  [first index, last index] for testing data handled by this job
        :param save_filenames: do step 1 (save the filenames)
        :param do_combine: do step 2 (combine the maps and store the maps)
        :param verbose: if True: print some information about each combined file
        """
        required_keys = ("gen", "data", "mod", "tt", "comb")
        self._check_keys_exist(required_keys)

        os.makedirs(self.p.gen["data_root"], exist_ok=True)
        os.makedirs(self.p.gen["combined_maps_root"], exist_ok=True)
        os.makedirs(self.p.gen["fermi_root"], exist_ok=True)

        # Check if data exists already
        if os.path.isdir(self.p.gen["combined_maps_folder"]):
            combined_folder_content = os.listdir(self.p.gen["combined_maps_folder"])
            if np.any(["params" in filename and filename.endswith(".pickle") for filename in combined_folder_content]):
                print("Combined maps exist already!")
                return

        # Step 1: save filenames of template maps to combine
        if save_filenames:
            combine_template_maps(True, self.p, job_id, train_range, val_range, test_range, verbose=verbose)

        # Step 2: combine
        if do_combine:
            combine_template_maps(False, self.p, job_id, train_range, val_range, test_range, verbose=verbose)

            # Save parameters
            param_subdict = get_subdict(self.p, keys=required_keys, return_normal_dict=True)
            datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
            with open(os.path.join(self.p.gen["combined_maps_root"],
                                   self.p.comb["data_name"] + "_" + str(self.p.data["nside"]),
                      "params_" + datetime + ".pickle"), "wb") as params_file:
                pickle.dump(param_subdict, params_file)

    def build_pipeline(self):
        """
        Build the input pipeline, consisting of generators and datasets
        """
        required_keys = ("gen", "data", "mod", "comb", "nn", "train")
        self._check_keys_exist(required_keys)
        for k in ["data_root", "models_root", "checkpoints_root", "summaries_root", "params_root", "figures_root"]:
            os.makedirs(self.p.gen[k], exist_ok=True)
        self.inds = build_index_dict(self.p)
        self.p.nn["input_shape"] = (len(self.inds["indexes"][0]), len(self.p.data["log_ebins"]) - 1)
        self.generators, self.datasets = build_pipeline(self.p)
        print("Input pipeline successfully built.")

    def build_nn(self):
        """
        Build neural network.
        """
        required_keys = ("gen", "data", "mod", "comb", "nn")
        self._check_keys_exist(required_keys)
        assert len(self.datasets) > 0 and len(self.generators) > 0, \
            "Data pipeline has not been built yet! Run 'build_pipeline()' first."

        # Get NN object
        if self.p.nn["NN_type"] == "CNN":
            self._nn_spec = DeepsphereCNN(self.p, self.inds, self.template_dict, strategy=self._strategy)
        else:
            raise NotImplementedError
        for k in ["checkpoints_folder", "checkpoints_folder_ff", "checkpoints_folder_hist",
                  "summaries_folder", "summaries_folder_ff", "summaries_folder_hist",
                  "params_folder", "figures_folder"]:
            os.makedirs(self.p.nn[k], exist_ok=True)

        # Build the NN
        self.nn, self._trainable_weights_dict = self._nn_spec.build_model()

    def train_nn(self, which, ff_means_only=False, last_layer_only=False, new_optimizer=False):
        """
        Train neural network
        :param which: submodel, must be one of "flux_fractions" or "histograms"
        :param ff_means_only: even if aleatoric uncertainties for flux fractions are enabled, only train the means
        :param last_layer_only: if True: only train the final fully-connected layer
        :param new_optimizer: if True: while weights will still be loaded from a checkpoint (if one is found), a new
        optimizer will be used. In particular, this means that the initial learning rate is restored, and the training
        restarts at global step 0.
        """
        required_keys = ("gen", "data", "mod", "comb", "nn", "train")
        self._check_keys_exist(required_keys)

        # Check that everything is correct
        which = which.lower()
        assert which in ["flux_fractions", "histograms"], "Choose which = 'flux_fractions' or 'histograms'!"
        assert len(self.datasets) > 0 and len(self.generators) > 0, \
            "Data pipeline has not been built yet! Run 'build_pipeline()' first."
        assert self.nn is not None, "Neural network has not been built yet! Run 'build_nn()' first."
        assert self.nn.built, "Neural network has not been built yet! Run 'build_nn()' first."
        if which == "flux_fractions":
            assert self.p.nn.ff["return_ff"], "self.p.nn.ff['return_ff'] is set to False!"
        elif which == "histograms":
            assert self.p.nn.hist["return_hist"], "self.p.nn.hist['return_hist'] is set to False!"
        if ff_means_only and which == "histograms":
            warnings.warn("Argument 'ff_means_only' is ignored for SCD histogram training!")

        # Define loss, learning rate scheduler
        loss, loss_keys = losses.get_loss_and_keys(which=which, params=self.p, ff_means_only=ff_means_only)
        lr_scheduler = getattr(tf.keras.optimizers.schedules,
                               self.p.train["scheduler"])(**self.p.train["scheduler_dict"])

        # Define metrics for evaluation
        metric_list = self.p.train["hist_train_metrics"] if which == "histograms" else self.p.train["ff_train_metrics"]
        metric_fct = losses.get_loss_and_keys_histograms if which == "histograms" \
            else losses.get_loss_and_keys_flux_fractions
        metric_keys = [metric_fct(metric)[1] for metric in metric_list]

        # Define the iterators
        train_iter = iter(self.datasets["train"].ds)
        val_iter = iter(self.datasets["val"].ds)
        label_ind = 1 if which == "histograms" else 0
        global_size_train = self.p.train["batch_size"]
        global_size_val = self.p.train["batch_size_val"]

        # if self.p.nn.hist["continuous"]:
        #     global_size_train *= self.p.train["hist_n_taus"] * self.p.train["hist_n_flux_queries"]
        #     global_size_val *= self.p.train["hist_n_taus"] * self.p.train["hist_n_flux_queries"]

        # Define file writer, checkpoint, and checkpoint managers
        summaries_folder_str = "summaries_folder_hist" if which == "histograms" else "summaries_folder_ff"
        summary_writer = tf.summary.create_file_writer(self.p.nn[summaries_folder_str])

        # Define optimizer, checkpoint, and managers for loading
        with self._strategy.scope():

            optimizer_load, checkpoint_load = self._get_new_optimizer_and_checkpoint(lr_scheduler)
            manager_ff_load = tf.train.CheckpointManager(checkpoint_load, directory=self.p.nn["checkpoints_folder_ff"],
                                                         max_to_keep=3)
            manager_hist_load = tf.train.CheckpointManager(checkpoint_load,
                                                           directory=self.p.nn["checkpoints_folder_hist"],
                                                           max_to_keep=3)

            # Check if training resumes or if training from scratch
            require_new_optimizer = True if new_optimizer else False

            # Flux fraction training
            if which == "flux_fractions":
                # Try to restore a flux fraction checkpoint
                checkpoint_load.restore(manager_ff_load.latest_checkpoint)
                if manager_ff_load.latest_checkpoint:
                    print("Restored checkpoint from {}.".format(manager_ff_load.latest_checkpoint))
                else:
                    print("Training flux fractions from scratch.")

                # in both cases: continue with the same optimizer, checkpoint, and manager
                optimizer, checkpoint, manager_write = optimizer_load, checkpoint_load, manager_ff_load
                self.nn.compile(optimizer=optimizer)

            # SCD histogram training
            elif which == "histograms":
                # Try to restore a histogram checkpoint
                checkpoint_load.restore(manager_hist_load.latest_checkpoint)
                if manager_hist_load.latest_checkpoint:
                    print("Restored checkpoint from {}.".format(manager_hist_load.latest_checkpoint))
                    # in this case: continue with the same optimizer, checkpoint, and manager
                    optimizer, checkpoint, manager_write = optimizer_load, checkpoint_load, manager_hist_load
                    self.nn.compile(optimizer=optimizer)
                else:
                    # Try to restore a flux fraction checkpoint
                    if manager_ff_load.latest_checkpoint:
                        checkpoint_load.restore(manager_ff_load.latest_checkpoint)
                        print("Restored flux fraction checkpoint from {}.".format(manager_ff_load.latest_checkpoint))
                    else:
                        if self.p.nn.hist["calculate_residual"]:
                            raise RuntimeError("Feeding the Poissonian residuals as a second channel for the SCD "
                                               "histogram estimation requires training the flux fractions first!")
                    print("Training SCD histograms from scratch.")
                    # in both cases: define a NEW optimizer and checkpoint
                    require_new_optimizer = True
            else:
                raise NotImplementedError

            # If new optimizer is required
            if require_new_optimizer:
                c_folder_str = "checkpoints_folder_hist" if which == "histograms" else "checkpoints_folder_ff"
                print("A new optimizer will be used for the training: initial learning rate is restored.")
                optimizer, checkpoint = self._get_new_optimizer_and_checkpoint(lr_scheduler)
                manager_write = tf.train.CheckpointManager(checkpoint, directory=self.p.nn[c_folder_str], max_to_keep=3)
                self.nn.compile(optimizer=optimizer)

        # Set trainable parameters:
        weights_to_train_key = "hist" if which == "histograms" else "ff"
        if last_layer_only:
            weights_to_train_key += "_final_dense"
        weights_to_train = self._trainable_weights_dict[weights_to_train_key]
        print("{:} tensor(s) will be trained.".format(len(weights_to_train)))

        # During training: quantile level tau is randomly drawn
        if self.p.nn.requires_tau:
            median_tau = 0.5 * tf.ones((self.p.train["batch_size"], 1))
            if which == "histograms":
                if self.p.train["hist_tau_prior"] == "uniform":
                    tau_dist = tfp.distributions.Uniform(low=0.0, high=1.0)
                elif self.p.train["hist_tau_prior"] == "uniform_in_z":
                    z_abs_max = 3.5
                    norm_dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
                    z_dist = tfp.distributions.Uniform(low=-z_abs_max, high=z_abs_max)
                else:
                    raise NotImplementedError(f"hist_tau_prior {self.p.train['hist_tau_prior']} unknown!")

                # if flux queries need to be drawn for the continuous case
                # if self.p.nn.hist["continuous"]:
                #     normed_flux_query_dist = tfp.distributions.Uniform(low=0.0, high=1.0)

        # NN input helper function that takes care of quantile levels tau during training
        def get_nn_input(data_):
            if self.p.nn.requires_tau:
                if which != "histograms":
                    tau = median_tau
                elif self.p.train["hist_tau_prior"] == "uniform":
                    tau = tau_dist.sample((data_.shape[0], 1))
                elif self.p.train["hist_tau_prior"] == "uniform_in_z":
                    z_vals = z_dist.sample((data_.shape[0], 1))
                    tau = norm_dist.cdf(z_vals)
                else:
                    raise NotImplementedError(f"hist_tau_prior {self.p.train['hist_tau_prior']} unknown!")
                nn_input = [data_, tau]
            else:
                nn_input = data_
            return nn_input

        # Loss helper function
        def get_loss(data_, label_, global_size, training):
            nn_input = get_nn_input(data_)
            nn_output = self.nn(nn_input, training=training)

            if which == "flux_fractions":
                if self.p.nn.ff["return_energies"]:
                    raise NotImplementedError
                else:
                    true_fluxes = tf.reduce_sum(label_, -1, keepdims=True)
                    true_flux_fractions = true_fluxes / tf.reduce_sum(true_fluxes, -2, keepdims=True)
                    return tf.reduce_sum(loss(true_flux_fractions, *[nn_output[k] for k in loss_keys])) / global_size

            else:
                return tf.reduce_sum(loss(label_, *[nn_output[k] for k in loss_keys]), 0) \
                       / global_size


            # multiplier = 1e0  # to avoid numerical issues
            # new_dict = {}
            #
            # # Convert flux fractions to fluxes
            # if which == "flux_fractions":
            #     total_fluxes_per_bin = tf.reduce_sum(label_, axis=1, keepdims=True)
            #     # total_fluxes = tf.reduce_sum(total_fluxes_per_bin, axis=2, keepdims=True)
            #     # ebin_weighting = total_fluxes_per_bin
            #     # new_dict["ff_mean"] = nn_output["ff_mean"] * total_fluxes_per_bin * multiplier
            #     new_dict["ff_mean"] = nn_output["ff_mean"] * multiplier
            #     if self.p.nn.ff["alea_var"]:
            #         new_dict["ff_logvar"] = nn_output["ff_logvar"] + 2 * tf.math.log(total_fluxes_per_bin) + tf.math.log(multiplier)  # log(x^2) = 2 * log(x)
            #
            #     label_ /= total_fluxes_per_bin
            #     # Average the loss over the energy bins
            #     return tf.reduce_mean(tf.reduce_sum(loss(label_ * multiplier, *[new_dict[k] for k in loss_keys]), 0)) \
            #               / global_size
            # else:
            #     return tf.reduce_sum(loss(label_, *[nn_output[k] for k in loss_keys]), 0) \
            #            / global_size

        # Define training step
        def train_step(data_, label_, global_size):
            with tf.GradientTape() as tape:
                loss_val_ = get_loss(data_, label_, global_size, training=True)
            gradients = tape.gradient(loss_val_, weights_to_train)
            optimizer.apply_gradients(zip(gradients, weights_to_train))
            return loss_val_

        # Define metrics evaluation step
        def get_metrics(data_, label_, global_size, training=False):
            metric_values = []
            nn_input = get_nn_input(data_)
            nn_output = self.nn(nn_input, training=training)

            if which == "flux_fractions":
                if self.p.nn.ff["return_energies"]:
                    raise NotImplementedError
                else:
                    true_fluxes = tf.reduce_sum(label_, -1, keepdims=True)
                    true_flux_fractions = true_fluxes / tf.reduce_sum(true_fluxes, -2, keepdims=True)
                    for metric, metric_keys_loc in zip(metric_list, metric_keys):
                        metric_values.append(tf.reduce_sum(metric_fct(metric)[0](true_flux_fractions, *[nn_output[k]
                                                                                                      for k in
                                                                                                      metric_keys_loc]),
                                                           ) / global_size)
            else:
                for metric, metric_keys_loc in zip(metric_list, metric_keys):
                    metric_values.append(tf.reduce_sum(metric_fct(metric)[0](label_, *[nn_output[k] for k in metric_keys_loc]),
                                                       0) / global_size)


            # multiplier = 1e8  # to avoid numerical issues
            # new_dict = {}
            #
            # # Convert flux fractions to fluxes
            # if which == "flux_fractions":
            #     total_fluxes_per_bin = tf.reduce_sum(data_, axis=1, keepdims=True)
            #     # total_fluxes = tf.reduce_sum(total_fluxes_per_bin, axis=2, keepdims=True)
            #     # ebin_weighting = total_fluxes_per_bin / total_fluxes
            #     new_dict["ff_mean"] = nn_output["ff_mean"] * multiplier
            #     if self.p.nn.ff["alea_var"]:
            #         new_dict["ff_logvar"] = nn_output["ff_logvar"] + 2 * tf.math.log(total_fluxes_per_bin) + tf.math.log(multiplier) # log(x^2) = 2 * log(x)
            #     for metric, metric_keys_loc in zip(metric_list, metric_keys):
            #         metric_values.append(tf.reduce_mean(tf.reduce_sum(metric_fct(metric)[0](label_, *[new_dict[k]
            #                                                                                           for k in
            #                                                                                           metric_keys_loc]),
            #                                                           0)) / global_size)
            # else:
            #     for metric, metric_keys_loc in zip(metric_list, metric_keys):
            #         metric_values.append(tf.reduce_sum(metric_fct(metric)[0](label_ * multiplier,
            #                                                                  *[nn_output[k] for k in metric_keys_loc]),
            #                                            0) / global_size)
            return metric_values

        # Wrapper around get_loss that takes care of the replicas in case multiple GPUs are available
        # NOTE: this function should NOT be used in distributed train step because gradients need to go inside strategy
        @tf.function
        def distributed_get_loss(data_, label_, global_size, training):
            per_replica_losses = self._strategy.run(get_loss, args=(data_, label_, global_size, training))
            return self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # Wrapper around train_step that takes care of the replicas in case multiple GPUs are available
        @tf.function
        def distributed_train_step(data_, label_, global_size):
            per_replica_losses = self._strategy.run(train_step, args=(data_, label_, global_size))
            return self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # Wrapper around get_metrics that takes care of the replicas in case multiple GPUs are available
        @tf.function
        def distributed_get_metrics(data_, label_, global_size, training=False):
            per_replica_metrics = self._strategy.run(get_metrics, args=(data_, label_, global_size, training))
            return [self._strategy.reduce(tf.distribute.ReduceOp.SUM, metric, axis=None)
                    for metric in per_replica_metrics]

        # Settings for training loop
        num_steps = self.p.train["num_steps"]
        eval_frequency = self.p.train["eval_frequency"]

        # Check if there is anything to be done
        steps_left = num_steps - optimizer.iterations.numpy()
        if steps_left <= 0:
            print("The total number of training steps ({:}) has already been reached!".format(num_steps))
            return

        # Save parameters
        param_subdict = get_subdict(self.p, keys=self.p.keys(), return_normal_dict=True, delete_functions=True)
        datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
        with open(os.path.join(self.p.nn["params_folder"], "params_" + datetime + "_" + which + ".pickle"), "wb") \
                as params_file:
            pickle.dump(param_subdict, params_file)

        # Save a plot of the neural network architecture
        self.plot_nn_architecture(filename="model_" + datetime + "_" + which + ".pdf")

        # Training loop
        print("\n=== STARTING TRAINING ===")
        with tqdm(enumerate(train_iter, start=0), total=steps_left) as pbar:
            for i_step, s in pbar:
                global_step = optimizer.iterations.numpy()
                data = s["data"]
                labels = s["label"][label_ind]
                loss_eval = distributed_train_step(data, labels, global_size=global_size_train)
                pbar.set_postfix(train_loss=loss_eval.numpy(), refresh=False)

                # Is it an evaluation step?
                if global_step % eval_frequency == 0:
                    # Write a checkpoint
                    manager_write.save(checkpoint_number=global_step)

                    # Evaluate
                    with summary_writer.as_default():
                        # TF behavior changed!
                        if int(tf.__version__.split(".")[1]) >= 11:
                            lr_summary = optimizer.learning_rate
                        else:
                            lr_summary = optimizer.learning_rate(global_step)
                        tf.summary.scalar('learning_rate', lr_summary, step=global_step)  # deleted (global_step)
                        tf.summary.scalar('losses/' + which + '/train_loss', loss_eval.numpy(), step=global_step)

                        if np.isnan(loss_eval.numpy()):
                            raise RuntimeError("NaN loss encountered during training!")

                        # Metrics on training data. NOTE: evaluating in "training = True" mode here
                        metric_train_eval = distributed_get_metrics(data, labels, global_size_train, training=True)
                        for i_metric, metric in enumerate(metric_list):
                            tf.summary.scalar("train_metrics/" + which + '/' + metric, metric_train_eval[i_metric],
                                              step=global_step)

                        # Loss & metrics on (a single) validation batch
                        val_samples = next(val_iter)
                        val_data = val_samples["data"]
                        val_labels = val_samples["label"][label_ind]
                        val_loss_eval = distributed_get_loss(val_data, val_labels, global_size_val, training=False)
                        tf.summary.scalar('losses/' + which + '/val_loss', val_loss_eval.numpy(), step=global_step)
                        metric_val_eval = distributed_get_metrics(val_data, val_labels, global_size_val,
                                                                  training=False)
                        for i_metric, metric in enumerate(metric_list):
                            tf.summary.scalar("val_metrics/" + which + '/' + metric, metric_val_eval[i_metric],
                                              step=global_step)

                # Reached end of training?
                if global_step >= num_steps:
                    print("=== TRAINING FINISHED ===\n")
                    break

    def load_nn(self, checkpoint_filename=None, load_from_ff_training=False):
        """
        This method loads the trained neural network. If no checkpoint is provided, it tries to restore the latest saved
        checkpoint. It first looks in the checkpoint folder for the histogram training and then for the flux fraction
        training.
        :param checkpoint_filename: the filename of the checkpoint file can be manually provided here. Otherwise, the
        latest checkpoint will be used.
        :param load_from_ff_training: if True, only look for checkpoint files in histogram checkpoint folder.
        """
        required_keys = ("nn", "train")
        self._check_keys_exist(required_keys)

        # Check that everything is correct
        assert len(self.datasets) > 0 and len(self.generators) > 0, \
            "Data pipeline has not been built yet! Run 'build_pipeline()' first."
        assert self.nn is not None, "Neural network has not been built yet! Run 'build_nn()' first."
        assert self.nn.built, "Neural network has not been built yet! Run 'build_nn()' first."

        # When using checkpoint instead of model.save_weights(): need to build optimizer before loading
        lr_scheduler = getattr(tf.keras.optimizers.schedules,
                               self.p.train["scheduler"])(**self.p.train["scheduler_dict"])

        with self._strategy.scope():
            optimizer, checkpoint = self._get_new_optimizer_and_checkpoint(lr_scheduler)

            if checkpoint_filename is None:
                subfolders = ["ff"] if load_from_ff_training else ["hist", "ff"]
                for sub in subfolders:
                    checkpoint_filename = tf.train.latest_checkpoint(self.p.nn["checkpoints_folder_" + sub])
                    if checkpoint_filename is not None:
                        break
                if checkpoint_filename is None:
                    raise FileNotFoundError("No checkpoint files found in {:}!".format(self.p.nn["checkpoints_folder"]))

            checkpoint.restore(checkpoint_filename)
        print("Checkpoint {:} successfully loaded.".format(checkpoint_filename))

    def _get_new_optimizer_and_checkpoint(self, lr_scheduler):
        """
        This method returns a new optimizer and checkpoint
        :param lr_scheduler: learning rate scheduler
        :return: optimizer, checkpoint
        """
        optimizer = getattr(tf.keras.optimizers, self.p.train["optimizer"])(lr_scheduler,
                                                                            **self.p.train["optimizer_dict"])
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.nn)
        return optimizer, checkpoint

    def predict(self, data, tau=None, normed_flux_queries=None, multiple_taus=False, compress=False, remove_exp=False,
                training=False):
        """
        This method is a wrapper around tf.keras.Model.call() that takes care of things such as the quantile levels and
        input shapes.
        :param data: numpy array or tensorflow tensor with photon-count map(s)
        :param tau: if not None: quantile levels of interest for SCD histograms
        :param normed_flux_queries: if flux queries are provided for continuous EMPL
        :param multiple_taus: if True, the neural network will be evaluated for ALL the values in tau (the same quantile
        levels are used for all the maps in data)
        :param compress: set this to True if full maps are provided to this function, they will then be compressed
        before being fed to the neural network
        :param remove_exp: if True: the maps will be scaled before feeding them to the neural network (same scaling as
        for the training data). NOTE: This requires self.p.nn['remove_exp'] = True
        :param training: will be passed to self.nn.call(...), sets training / testing behavior for batch norm etc.
        :return: dictionary containing the NN predictions for data
        """
        required_keys = ("gen", "data", "mod", "comb", "nn")
        self._check_keys_exist(required_keys)

        assert len(self.datasets) > 0 and len(self.generators) > 0, \
            "Data pipeline has not been built yet! Run 'build_pipeline()' first."

        if remove_exp:
            assert self.p.nn["remove_exp"], "When self.p.nn['remove_exp'] = False, remove_exp also must be False!"

        is_tf = tf.is_tensor(data)
        if not is_tf:
            assert isinstance(data, np.ndarray), "Data must be TF tensor or numpy array!"

        # if no batch dimension is provided, add now
        if len(data.shape) == 1:
            exp_dims = tf.expand_dims if is_tf else np.expand_dims
            data = exp_dims(data, 0)

        # compress?
        if compress:
            data = self.compress(data)

        # rescale?
        if remove_exp:
            data /= self.generators["train"].settings_dict["rescale"][None]

        # now: take care of quantile levels tau
        if self.p.nn.requires_tau:
            if tau is None:
                print("No values of tau have been provided. They will be randomly drawn from U(0, 1).")
                unif = np.random.uniform(low=0.0, high=1.0, size=[data.shape[0], 1]).astype(np.float32)
                nn_input = [data, unif]
            else:
                if np.isscalar(tau):
                    tau = tau * np.ones((data.shape[0], 1))
                # if self.p.nn.hist["continuous"]:
                #     assert normed_flux_queries is not None, "Flux queries must be provided for continuous EMPL!"
                #     nn_input = [data, tau, normed_flux_queries]
                # else:
                nn_input = [data, tau]
        else:
            nn_input = data
            if tau is not None:
                warnings.warn("A value of tau has been provided; however, SCD histogram estimation with the EMPL is "
                              "disabled. The values of tau will be ignored.")
            if multiple_taus:
                warnings.warn("SCD histogram estimation with the EMPL is disabled, 'multiple_taus' will be ignored.")

        if tau is None and multiple_taus:
            warnings.warn("'multiple_taus' is True, but since no quantile levels tau have been provided, "
                          "'multiple_taus' will be ignored.")

        # Return NN prediction
        if self.p.nn.requires_tau and multiple_taus:
            outdict = {}
            for i_ql, ql in enumerate(tau):
                this_ql = ql * np.ones((data.shape[0], 1))
                # if self.p.nn.hist["continuous"]:
                #     this_outdict = self.nn([data, this_ql, normed_flux_queries], training=training)
                # else:
                this_outdict = self.nn([data, this_ql], training=training)

                for k in this_outdict.keys():
                    if k in ["tau", "hist", "f_query"]:
                        if k not in outdict:
                            outdict[k] = tf.expand_dims(this_outdict[k], 0)
                        else:
                            outdict[k] = tf.concat([outdict[k], tf.expand_dims(this_outdict[k], 0)], 0)
                    else:
                        if k not in outdict:
                            outdict[k] = this_outdict[k]
            return outdict
        else:
            return self.nn(nn_input, training=training)

    def delete_run(self, confirm=True):
        """
        Empties the checkpoint, summary, parameter, and figures folders belonging to a training run.
        :param confirm: if True: user needs to confirm
        """
        required_keys = ("nn",)
        self._check_keys_exist(required_keys)

        run_name = self.p.nn["run_name"]
        if confirm:
            user_input = input("Confirm deletion of run '{:}' with 'Y':".format(run_name))
            if user_input != "Y":
                print("Aborting...")
                return

        folders = [self.p.nn[k] for k in ["params_folder", "figures_folder",
                                          "checkpoints_folder_ff", "checkpoints_folder_hist",
                                          "summaries_folder_ff", "summaries_folder_hist"]]
        for folder in folders:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        print("Deleted run '{:}'.".format(run_name))

    def plot_nn_architecture(self, filename="model.pdf"):
        """
        Make a plot of the neural network architecture
        """
        required_keys = ("nn",)
        self._check_keys_exist(required_keys)

        # Save a flowchart to the figures folder
        # The following requires pydot and graphviz to be installed
        try:
            tf.keras.utils.plot_model(self.nn, show_shapes=True, to_file=os.path.join(self.p.nn["figures_folder"],
                                                                                      filename))
        except AssertionError:
            pass  # in case plotting doesn't work

    def plot_flux_fractions(self, true_ffs, preds, **kwargs):
        """
        Plot true vs. estimated flux fractions.
        :param true_ffs: true flux fractions
        :param preds: neural network prediction (output dictionary)
        :param kwargs: will be passed on to plot_flux_fractions() in plots.py
        :return: figure, axes
        """
        required_keys = ("mod", "nn", "plot")
        self._check_keys_exist(required_keys)
        assert self.p.nn.ff["return_ff"], "self.p.nn.ff['return_ff'] is set to False!"
        return plot_flux_fractions(self.p, true_ffs, preds, **kwargs)

    def plot_histograms(self, true_hists, preds, **kwargs):
        """
        Plot (true and) estimated histograms.
        :param true_hists: true flux fractions
        :param preds: neural network prediction (output dictionary)
        :param kwargs: will be passed on to plot_histograms() in plots.py
        :return: list of figures, list of axes (elements correspond to histogram templates)
        """
        required_keys = ("mod", "data", "nn", "plot")
        self._check_keys_exist(required_keys)
        assert self.p.nn.hist["return_hist"], "self.p.nn.hist['return_hist'] is set to False!"
        return plot_histograms(self.p, true_hists, preds, mean_exp=self.template_dict["mean_exp_roi"], **kwargs)

    def plot_maps(self, maps, decompress=True, **kwargs):
        """
        Plot maps (produced by the dataset objects).
        :param maps: maps (compressed or decompressed).
        :param decompress: set to True if maps are compressed and need to be decompressed
        :param kwargs: will be passed on to plot_maps() in plots.py
        :return: figure, axes
        """
        required_keys = ("data", "nn")
        self._check_keys_exist(required_keys)
        assert len(maps.shape) == 2, "Shape of maps must be n_maps x n_pixels."
        if decompress:
            maps = self.decompress(maps, fill_value=np.nan)
        plot_maps(maps, self.p, **kwargs)

    def decompress(self, m, fill_value=0.0):
        """
        Decompresses a map (or batch of maps).
        :param m: single compressed map, or batch of compressed maps
        :param fill_value: fill value outside ROI
        :return: full map(s).
        """
        assert len(self.inds) > 0, "Pipeline must be built before 'decompress()' is available!"
        if isinstance(m, np.ndarray):
            if len(m.shape) <= 2:
                out = masked_to_full(m, self.inds['indexes'][0], nside=self.p.data["nside"], fill_value=fill_value)
            else:
                raise NotImplementedError("len(m.shape) > 2 not implemented!")
        else:
            raise TypeError("Expected numpy array, but found {:}.".format(type(m)))
        return out

    def compress(self, m):
        """
        Compresses a full map (or batch of maps).
        :param m: single full map, or batch of full maps
        :return: compressed map(s).
        """
        assert len(self.inds) > 0, "Pipeline must be built before 'compress()' is available!"
        if isinstance(m, np.ndarray):
            if len(m.shape) <= 2:
                assert hp.isnpixok(m.shape[-1]), "Expected full map(s) at npix = {:}, but found shape: {:}.".format(
                    self.p.data["npix"], m.shape)
                out = m[..., self.inds['indexes'][0]]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def set_plot_defaults():
        sns.set_context("talk")
        sns.set_style("ticks")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.rcParams["font.size"] = 14
        plt.rc('xtick', labelsize='small')
        plt.rc('ytick', labelsize='small')
