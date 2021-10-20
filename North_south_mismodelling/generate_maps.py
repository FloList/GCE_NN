"""
Generate I_1 (north-south asymmetry) and I_2 (pixels are randomly shuffled).
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from NPTFit import create_mask as cm  # Module for creating masks
import colorcet as cc
import seaborn as sns


def set_NS_template(A_N, A_S, nside=128):
    """Build a template with different normalisations in the two hemispheres"""
    npix = hp.nside2npix(nside)
    template = np.empty(npix)
    pix_N = np.argwhere(hp.pix2vec(nside, range(npix))[-1] >= 0).flatten()
    pix_S = np.argwhere(hp.pix2vec(nside, range(npix))[-1] < 0).flatten()
    template[pix_N] = A_N
    template[pix_S] = A_S
    return template


# Load template and exposure maps
A_N, A_S = 10, 1
name = "NS_asymmetry"  # Name of output file
nside = 128
npix = hp.nside2npix(nside)
outer_ring = 25
inner_band = 0
raw_map = np.ones(npix)
temp = set_NS_template(A_N, A_S, nside)
total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band, mask_ring=True, inner=0,
                                    outer=outer_ring, nside=nside)
temp *= (1 - total_mask_neg)
hp.mollview(temp)
temp_compressed = temp.copy()
compression_inds = np.argwhere(temp_compressed > 0)[:, 0]
compress_fn = lambda x: x[compression_inds]


def decompress_fn(x):
    raw_map = np.zeros(npix)
    raw_map[compression_inds] = x
    return raw_map


temp_compressed = compress_fn(temp)

# Generate a Poissonian realisation
I_1_compressed = np.random.poisson(temp_compressed)
I_1 = decompress_fn(I_1_compressed)

# Get a random permutation
sigma = np.random.permutation(len(I_1_compressed))
I_2_compressed = I_1_compressed[sigma]
I_2 = decompress_fn(I_2_compressed)

# Plot the two maps
# cmap = cc.cm.bgy
cmap = sns.cm.rocket_r
I_1_plot, I_2_plot = I_1.copy(), I_2.copy()
I_1_plot[total_mask_neg.astype(bool)] = np.nan
I_2_plot[total_mask_neg.astype(bool)] = np.nan
badcol = [0, 0, 0, 0]


def zoom_ax(ax):
    ax.set_xlim([-outer_ring - 1, outer_ring + 1])
    ax.set_ylim([-outer_ring - 1, outer_ring + 1])


hp.cartview(I_1_plot, title="North-south asymmetry", cmap=cmap, badcolor=badcol)
zoom_ax(plt.gca())
fig = plt.gcf()
ax = plt.gca()
rect = patches.Rectangle((-15, -15), 30, 30, linewidth=2, edgecolor='white', facecolor='none')
ax.add_patch(rect)
fig.savefig("North-south_asymmetry_rocket.pdf")

hp.cartview(I_2_plot, title="Randomly shuffled", cmap=cmap, badcolor=badcol)
zoom_ax(plt.gca())
fig = plt.gcf()
ax = plt.gca()
rect = patches.Rectangle((-15, -15), 30, 30, linewidth=2, edgecolor='white', facecolor='none')
ax.add_patch(rect)
fig.savefig("Randomly_shuffled_rocket.pdf")

# Save
save_folder = "/home/flo/PycharmProjects/GCE/DeepSphere/North_south_mismodelling/Data"
save_filename = os.path.join(save_folder, "data")
np.savez(save_filename, I_1=I_1, I_2=I_2, sigma=sigma, compression_inds=compression_inds)

# Load
# dl = np.load(save_filename + ".npz")
# I_1, I_2, sigma, compression_inds = dl["I_1"], dl["I_2"], dl["sigma"], dl["compression_inds"]
