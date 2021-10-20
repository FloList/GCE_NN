"""
Analyse maps I_1 and I_2 with NPTFit plot the results.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
from NPTFit import nptfit
from NPTFit import create_mask as cm
from NPTFit import dnds_analysis
from gce_utils import get_is_HPC
import seaborn as sns
sns.set_style("ticks")

HPC = get_is_HPC()
if HPC > 1:
    raise NotImplementedError

compress_fn = lambda x: x[compression_inds]
def decompress_fn(x, npix=hp.nside2npix(128)):
    raw_map = np.zeros(npix)
    raw_map[compression_inds] = x
    return raw_map

# Load data
if HPC:
    save_folder = "/scratch/u95/fl9575/GCE_v2/North_south_mismodelling/Data"
else:
    save_folder = "/home/flo/PycharmProjects/GCE/DeepSphere/North_south_mismodelling/Data"
save_filename = os.path.join(save_folder, "data")
dl = np.load(save_filename + ".npz")
I_1, I_2, sigma, compression_inds = dl["I_1"], dl["I_2"], dl["sigma"], dl["compression_inds"]

# Settings
DO_FIT = False
nside = 128
npix = hp.nside2npix(nside)
outer_ring = 25
inner_band = 0
raw_map = np.ones(npix)
temp_iso = raw_map.copy()
total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band,
                                    mask_ring=True, inner=0, outer=outer_ring, nside=nside)
temp_iso *= (1 - total_mask_neg)
# hp.mollview(temp_iso)


for sim, tag in zip([I_1, I_2], ['NS_asymmetry', 'NS_shuffled']):
    np.random.seed(0)
    print("Starting with", tag, "...")

    # Set the map and exposure
    exp = np.ones_like(sim)

    # Initialise NPTFit
    n = nptfit.NPTF(tag=tag)
    n.load_data(sim.astype(np.int32), exp)

    # Add mask
    n.load_mask(total_mask_neg)

    # Add the templates
    n.add_template(temp_iso, 'iso')
    n.add_template(temp_iso, 'iso_np', units='PS')

    # Add the Poisson model
    n.add_poiss_model("iso", '$A_\mathrm{iso}$', [-6, 2.], True)

    # Add the Non-poisson model
    n.add_non_poiss_model('iso_np',
                          ['$A_\mathrm{iso}^\mathrm{ps}$', '$n_1^\mathrm{iso}$', '$n_2^\mathrm{iso}$',
                           '$S_b^{\mathrm{iso}}$'],
                          [[-6., 2.], [2.05, 30], [-30, 1.95], [0.05, 40.0]],
                          [True, False, False, False])


    # No PSF correction in this example

    # Configure and run the scan
    n.configure_for_scan(nexp=1)

    if DO_FIT:
        n.perform_scan(nlive=500)
    else:
        # Analyse!
        n.load_scan()
        an = dnds_analysis.Analysis(n)
        plt.ion()
        an.make_triangle()
        fig = plt.gcf()
        fig.savefig("Triangle_NPTFit_" + tag + ".pdf")

        # Plot flux fractions
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(3, 3))
        colour_P = 'deepskyblue'
        colour_NP = 'darkslateblue'
        nbins = 1500
        an.plot_intensity_fraction_poiss("iso", bins=nbins, color=colour_P, label="P", lw=3)
        an.plot_intensity_fraction_non_poiss("iso_np", bins=nbins, color=colour_NP, label="NP", lw=3)
        ax_1.set_xlabel('Flux fraction (%)')
        ax_1.legend(fancybox=True)
        ax_1.set_xlim(0, 100)
        ax_1.set_ylim(0, .1)
        plt.tight_layout()
        fig_1.savefig("FF_NPTFit_" + tag + ".pdf")
