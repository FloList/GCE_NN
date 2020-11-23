"""
This script can be used to plot dN/dF source count distribution functions, given by broken power laws (see NPTFit source
code).
"""
######################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os
from gce_utils import *
from NPTFit import create_mask as cm  # Module for creating masks

# Settings
T_NP = "gce_12_PS"
fermi_folder = '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'
CONST_EXP = True
save_fig = False
total_mask_neg = cm.make_mask_total(nside=128, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=25)

A_0 = 0.15
S_0 = 1.3189
scale_fac = 0.82

# model_params = [[-0.71, 33.97, -0.93, 3.75], [-0.77, 34.23, -0.93, 4.28], [-1.15, 33.89, -1.05, 7.13], [-1.46, 34.34, -1.08, 9.52]]  # GCE_for_letter best fits within 25*, 20*, 15*, 10*
# model_params = [[-2.8, 9.5, -1, 15]]  # this one is similar to the hard one in Chang et al. 2020
# model_params = [[-3.5, 10, 1.9, -0.8, 22, 0.2]]  # this one is similar to the soft one in Chang et al. 2020
# model_params = [[-2.8, 9.5, -1, 15], [-3.5, 10, 1.9, -0.8, 22, 0.2], [-0.71, 33.97, -0.93, 3.75]]  # hard, soft, mine from GCE_for_letter
# model_params = [[10.5, 10, -1.2, 1.3189], [8.5, 10, -1.2, 13.189], [6.5, 10, -1.2, 131.89]]   # dim, default, bright, from Nick. NOTE: A is in terms of flux!
# model_params = [[A_0, 10, -1.2, S_0], [A_0-2*scale_fac, 10, -1.2, S_0*10**scale_fac], [A_0-4*scale_fac, 10, -1.2, S_0*10**(2*scale_fac)]]   # dim, default, bright, my adaptation
model_params = [[A_0, 10, -1.2, S_0], [A_0-scale_fac, 10, -1.2, S_0*10**(0.5 * scale_fac)], [A_0-2*scale_fac, 10, -1.2, S_0*10**scale_fac],
                [A_0-3*scale_fac, 10, -1.2, S_0*10**(1.5 * scale_fac)], [A_0-4*scale_fac, 10, -1.2, S_0*10**(2*scale_fac)]]   # dim, def-dim, default, def-bright, bright, my adaptation

# Add parameters for best-fit mocks
model_params += [[-1.46, 34.34, -1.08, 9.52], [-0.71, 33.97, -0.93, 3.75]]

# Load exposure map
exp = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
mean_exp = np.mean(exp)
const_exp_map = np.ones_like(exp) * mean_exp
cor_term = np.log10(mean_exp)
rescale = exp / mean_exp
exp_for_data_generation = const_exp_map if CONST_EXP else exp
exp_for_data_generation_masked = exp_for_data_generation * (1 - total_mask_neg)

# Load template
template = get_template(fermi_folder, T_NP[:-3])
template /= rescale  # remove Fermi exposure correction from template
template *= (1 - total_mask_neg)  # apply mask

template_sum = template.sum()
pixarea = hp.nside2pixarea(128)
smin = 0.01
smax = 1000
nsteps = 1000
# Set an array of counts
sarray = 10 ** np.linspace(np.log10(smin), np.log10(smax), nsteps)
# Convert to flux
flux_array = sarray / exp_for_data_generation.mean()

# Set up plot and total flux array
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
total_flux_array = np.zeros((len(model_params)))
total_counts_array = np.zeros((len(model_params)))

# styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
colours = ['#ca0020', '#f4a582', '.7', '#92c5de', '#0571b0'][::-1]
colours += ["#756bb1", "#c51b8a"]

styles = ["-"] * 5
lw = [3] * 5 + [2] * 5
alphas_fill = [0.1, 0.05]

for i_theta, theta in enumerate(model_params):
    # Get dnds
    data_array = dnds(theta, sarray, a_is_log=True)
    # Determine area of the mask in deg^2
    area_mask = np.sum(1 - total_mask_neg) * pixarea * (360 / (2.*np.pi)) ** 2
    # Rescaling factor to convert dN/dS to [(ph /cm^2 /s)^-2 /deg^2]
    rf = template_sum * exp_for_data_generation.mean() / area_mask
    if i_theta < len(model_params) - 2:
        ax.plot(flux_array, rf * flux_array ** 2 * data_array, lw=lw[i_theta], color=colours[i_theta], ls=styles[i_theta])
    else:
        ax.fill_between(flux_array, rf * flux_array ** 2 * data_array, lw=lw[i_theta], color=colours[i_theta], zorder=-1, alpha=alphas_fill[i_theta - 5])
    total_flux_array[i_theta] = return_intensity_arrays_non_poiss(theta, template, exp_for_data_generation, counts=False, nside=128, a_is_log=True)
    total_counts_array[i_theta] = return_intensity_arrays_non_poiss(theta, template, exp_for_data_generation, counts=True, nside=128, a_is_log=True)
    plt.show()

ax.set_xscale("log")
ax.set_yscale("log")
xlim = np.asarray([2e-13, 8e-9])
ylim = np.asarray([5e-16, 1e-9])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.tick_params(axis='x', length=5, width=2, labelsize=18)
ax.tick_params(axis='y', length=5, width=2, labelsize=18)
ax.set_ylabel('$F^2 \, dN/dF$ [counts cm$^{-2}$ s$^{-1}$ deg$^{-2}$]', fontsize=18)
ax.set_xlabel('$F$  [counts cm$^{-2}$ s$^{-1}$]', fontsize=18)
plt.tight_layout()
# Plot 1 photon line
flux_1_ph = 1 / exp_for_data_generation.mean()
ax.axvline(flux_1_ph, linestyle="--", color="0.5")
ax.text(x=0.9 * flux_1_ph, y=3e-10, s="1 ph", rotation="horizontal", horizontalalignment='right',
        verticalalignment='bottom', fontsize=14, color="k")
# Plot Fermi detection threshold
rect = mpl.patches.Rectangle((4e-10, ylim[0]), 1e-10, ylim[1]-ylim[0], linewidth=0, edgecolor=None, facecolor="#ffeda0")
ax.text(x=0.9 * 4e-10, y=3e-10, s="3FGL", rotation="horizontal", horizontalalignment='right',
        verticalalignment='bottom', fontsize=14)
ax.add_patch(rect)
plt.show()
pretty_plots()

print("Total fluxes:\n", total_flux_array)
print("Total counts:\n", total_counts_array)

# Add Bartels dN/dF
bartels_path = "/home/flo/Documents/Latex/GCE/Data_for_paper/Plots_for_revision/Data/dNdF_MSP_Bartels.npz"
bartels = np.load(bartels_path)
F_centers_ary, dNdF_ary = bartels["F_centers_ary"], 0.3879197934308049 * bartels["dNdF_ary"]  # rescale to get the correct FF
dNdF_ary[F_centers_ary < 1. / mean_exp] = 0.0

ax.plot(F_centers_ary[dNdF_ary > 0], (dNdF_ary * F_centers_ary ** 2)[dNdF_ary > 0], color="forestgreen", ls="-.")
# plt.figure(); plt.plot(F_centers_ary, dNdF_ary * F_centers_ary ** 2)
# plt.xscale("log")
# plt.yscale("log")

# Calculate flux
template_mean = np.nanmean(template / exp_for_data_generation_masked / pixarea)
s_centers_ary = F_centers_ary * mean_exp
ds = [s_centers_ary[i + 1] - s_centers_ary[i] for i in range(len(s_centers_ary)-1)]
ds = np.array(ds + [ds[-1]])
dNdS_ary = dNdF_ary / mean_exp
intensity_bartels = np.sum(template_mean * dNdS_ary * s_centers_ary * ds)
print(intensity_bartels)

# Save
if save_fig:
    fig.savefig("SCDs_with_best_fit.pdf", bbox_inches="tight")
