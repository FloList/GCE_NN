"""
This file can be used to compare the theoretically expected flux fractions (from return_intensity_arrays_(non)_poiss in
gce_utils.py) to the actual flux fractions in the realisations (or to the NN predictions).
The example data used for this test is the one from GCE_and_background_const_exp.
"""
import numpy as np
import healpy as hp
import os
import pickle
from gce_utils import *
import time
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from NPTFit import create_mask as cm  # Module for creating masks
sns.set_style("white")
sns.set_context("talk")
######################################################
# Compare real or NN estimates to expected values?
take_NN_estimates = True

# Settings
fermi_folder = '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'
outer_ring = 25.0
nside = 128

MODELS = ["dif_O_pibs", "dif_O_ic", "iso", "bub", "gce_12", "gce_12_PS"]  # order needs to be the same as saved in NN_dict!
model_names = [r"diffuse $\pi^0$ + BS", "diffuse IC", "isotropic", r"$\it{Fermi}$ bubbles", "GCE DM", "GCE PS"]

NN_file = "/home/flo/Documents/Latex/GCE/Data_for_paper/GCE_and_background_12_const_exp/GCE_and_background_NN_predictions.npz"
settings_file = "/home/flo/Documents/Latex/GCE/Data_for_paper/GCE_and_background_12_const_exp/GCE_and_background_settings.pickle"
data_file = "/home/flo/Documents/Latex/GCE/Data_for_paper/GCE_and_background_12_const_exp/GCE_and_background_subset_256.pickle"

# Load raw data file
data_pickle = open(data_file, 'rb')
data_dict = pickle.load(data_pickle)
data_pickle.close()

# Load settings file
settings_pickle = open(settings_file, 'rb')
settings_dict = pickle.load(settings_pickle)
settings_pickle.close()

# Load NN_file
NN_dict = np.load(NN_file)
real_fluxes = NN_dict["real_fluxes"]
pred_fluxes = NN_dict["pred_fluxes"]
if take_NN_estimates:
    fluxes_to_compare = pred_fluxes
else:
    fluxes_to_compare = real_fluxes

# Get template names (P / NP)
T_all = np.asarray(list(data_dict["info"]["A"].keys()))
T_NP = np.asarray(list(data_dict["info"]["n"].keys()))
T_P = np.setdiff1d(T_all, T_NP)

# Load templates and remove Fermi exposure correction
exp_fermi = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
mean_exp_fermi = np.mean(exp_fermi)
rescale_fermi = exp_fermi / mean_exp_fermi

# Get templates and mask them
total_mask_neg = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=outer_ring)
T_dict = dict()
for temp in T_P:
    T_dict[temp] = get_template(fermi_folder, temp) / rescale_fermi * (1 - total_mask_neg)
for temp in T_NP:
    T_dict[temp] = get_template(fermi_folder, temp[:-3]) / rescale_fermi * (1 - total_mask_neg)

# Get masked exposure map that corresponds to the
exp_masked = settings_dict["exp"] * (1 - total_mask_neg)

# Compare fluxes to theoretically expected fluxes
n_samples = real_fluxes.shape[0]
flux_fraction_dict = dict()
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (list, np.ndarray)) else (a,)))
for i_sample in range(n_samples):
    flux_dict = dict()
    for model in T_P:
        flux_dict[model] = return_intensity_arrays_poiss(data_dict["info"]["A"][model][i_sample], T_dict[model], exp_masked, counts=False, nside=128,
                                                         a_is_log=True)
    for model in T_NP:
        theta = list(flatten([data_dict["info"]["A"][model][i_sample], data_dict["info"]["n"][model][i_sample], data_dict["info"]["S"][model][i_sample]]))
        flux_dict[model] = return_intensity_arrays_non_poiss(theta, T_dict[model], exp_masked, counts=False, nside=128,
                                                             a_is_log=True)

    total_flux = np.asarray([v for k, v in flux_dict.items()]).sum(0)
    for model in T_all:
        if model not in flux_fraction_dict.keys():
            flux_fraction_dict[model] = list()
        flux_fraction_dict[model].append(flux_dict[model] / total_flux)

# Make a plot
n_col = max(int(np.ceil(len(MODELS) // 2)), 1)
n_row = int(np.ceil(len(MODELS) / n_col))
scat_fig, scat_ax = plt.subplots(n_row, n_col, figsize=(10, 10), squeeze=False, sharex="all", sharey="all")
analyt_fluxes = np.asarray([flux_fraction_dict[key] for key in MODELS]).T

for i_ax, ax in enumerate(scat_ax.flatten(), start=0):
    if i_ax >= len(MODELS):
        continue
    ax.plot([0, 1], [0, 1], 'k-', lw=2, alpha=0.5)
    ax.fill_between([0, 1], y1=[0.05, 1.05], y2=[-0.05, 0.95], color="0.6", alpha=0.5)
    ax.fill_between([0, 1], y1=[0.1, 1.1], y2=[-0.1, 0.9], color="0.8", alpha=0.5)

for i_ax, ax in enumerate(scat_ax.flatten(), start=0):
    if i_ax >= len(MODELS):
        continue
    ax.scatter(analyt_fluxes[:, i_ax], fluxes_to_compare[:, i_ax], s=4, c="k")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Expected")
    if take_NN_estimates:
        ax.set_ylabel("Estimated flux fractions")
    else:
        ax.set_ylabel("True flux fractions")
    ax.set_title(model_names[i_ax])
plt.tight_layout()
plt.show()
pretty_plots()
