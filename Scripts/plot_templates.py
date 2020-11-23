"""
This script plots the Fermi templates and counts.
"""
import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from gce_utils import get_template, pretty_plots
plt.ion()
plt.rcParams['figure.figsize'] = (8, 7)
import seaborn as sns
sns.set_style("white")
# sns.set_context("talk")
plt.rcParams['image.cmap'] = 'rocket'
from NPTFit import create_mask as cm  # Module for creating masks

fermi_folder = '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'

fermi_exposure = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
rescale = fermi_exposure / fermi_exposure.mean()

templates_to_plot = ["dif_O_pibs", "dif_O_ic", "dif_A_pibs", "dif_A_ic", "dif_F_pibs", "dif_A_ic", "dif", "iso", "bub", "gce_12", "thin_disk", "thick_disk"]
# templates_to_plot = ["bub", "bub_var"]
model_names = [r"O: diffuse $\pi^0$ + BS", r"O: diffuse IC", r"A: diffuse $\pi^0$ + BS", r"A: diffuse IC",
               r"F: diffuse $\pi^0$ + BS", r"F: diffuse IC", r"diffuse p6v11", r"isotropic", r"$\it{Fermi}$ bubbles",
               r"GCE", r"thin disk", r"thick disk"]
# model_names = ["Bubbles training", "Bubbles evaluation"]
total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=25)

fig, axs = plt.subplots(2, len(templates_to_plot) // 2, frameon=False)
for i_temp, temp in enumerate(templates_to_plot):
    t_map = get_template(fermi_folder, temp)
    t_map /= rescale
    t_map[np.argwhere(total_mask_neg).flatten()] = 0.0
    hp.mollview(t_map, fig=fig, title=model_names[i_temp], cbar=False, notext=True, sub=(2, len(templates_to_plot) // 2, i_temp+1))
    ax = axs.flatten()[i_temp]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis("off")
    mw_ax = plt.gca()
    mw_ax.set_xlim([-0.3, 0.3])
    mw_ax.set_ylim([-0.4, 0.4])
pretty_plots()

# Fermi exposure map
exp = get_template(fermi_folder, "exp")
exp[np.argwhere(total_mask_neg).flatten()] = 0.0
hp.mollview(exp, title=r"$\it{Fermi}$ exposure", cbar=True)
mw_ax = plt.gca()
mw_ax.set_xlim([-0.3, 0.3])
mw_ax.set_ylim([-0.4, 0.4])
pretty_plots()
