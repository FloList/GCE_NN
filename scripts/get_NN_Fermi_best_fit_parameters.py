"""
Determine priors such that total fluxes and flux fractions give the NN best-fit prediction
"""
import numpy as np
import healpy as hp
import os
from NPTFit import create_mask as cm  # Module for creating masks
from GCE.data_utils import get_template
######################################################
TASK = "GCE_edep"
DEBUG = True
######################################
JOB_ID = 0
print("JOB IS is", JOB_ID, ".\n")
######################################
base_data_folder = "/home/flo/Documents/Projects/GCE_hist/GCE_NN/data"

# Load Florian Wolf's data
folder_florian_wolf = os.path.join(base_data_folder, "Florian_Wolfs_results")
energies = np.load(os.path.join(folder_florian_wolf, "x_axis.npy"))
E_sqr_dNdE = np.load(os.path.join(folder_florian_wolf, "y_axis_e_sq.npy"))
dNdE = E_sqr_dNdE / energies ** 2
flux_fractions = dNdE / dNdE.sum(0, keepdims=True)

# Settings
e_inds = 10, 20
outer_ring = 25.0
name = "best_fit"
nside = 256
npix = hp.nside2npix(nside)
fermi_folder = os.path.join(base_data_folder, f"fermi_data_edep/fermi_data_{nside}")


#############
# CHOOSE MODELS
if TASK == "GCE_edep":
    T_P = ["dif_O_pibs", "dif_O_ic", "iso", "bub"]
    T_NP = ["gce_12_PS", "thin_disk_PS"]
else:
    raise NotImplementedError
#############

# Exposure map
exp = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))[e_inds[0]:e_inds[1]]
mean_exp = np.mean(exp)
cor_term = np.log10(mean_exp)
rescale = exp / mean_exp
fermi_data = np.load(os.path.join(fermi_folder, 'fermidata_counts.npy'))[e_inds[0]:e_inds[1]]

# Set up the mask
total_mask_neg = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=outer_ring)
pscmask = np.load(os.path.join(fermi_folder, "fermidata_pscmask_3fgl.npy"))
total_mask_neg = (1 - (1 - total_mask_neg) * (1 - pscmask[e_inds[0]:e_inds[1]])).astype(bool)
ebins = 2 * np.logspace(-1, 3, 41)

# Load the templates
T_dict, T_corr_dict = dict(), dict()
for temp in T_P:
    if temp in ["iso", "bub"]:
        smooth = True
    else:
        smooth = False
    if "dif" not in temp:
        T_dict[temp] = get_template(fermi_folder, temp, smooth)[e_inds[0]:e_inds[1]]
    else:
        T_dict[temp] = get_template(fermi_folder, temp, smooth)
for temp in T_NP:
    poiss_temp = get_template(fermi_folder, temp[:-3])[e_inds[0]:e_inds[1]]
    T_dict[temp[:-3]] = T_dict[temp] = poiss_temp

indices = [np.argwhere(~m).flatten() for m in total_mask_neg]
template_sums_P = [None] * len(T_P)

for i_t, temp in enumerate(T_P):
    template_sums_P[i_t] = np.asarray([T_dict[temp][i, inds].sum() for (i, inds) in enumerate(indices)])

template_sums_P = np.asarray(template_sums_P)

# Fermi counts per template
fermi_counts_per_temp = fermi_data.sum(1, keepdims=True) * flux_fractions.T

# Resulting A's for Poissonian templates
A = fermi_counts_per_temp[:, :len(T_P)] / template_sums_P.T
print(A)

# PS models: fluxes for "flux_lims" parameter
fermi_flux_in_roi = np.asarray([(fermi_data[i, inds] / exp[i, inds]).sum() for (i, inds) in enumerate(indices)])
fluxes_PS = flux_fractions[len(T_P):, :] * fermi_flux_in_roi
print(fluxes_PS.mean(1))




