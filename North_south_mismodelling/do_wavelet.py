"""
Convolve maps I_1 and I_2 with Mexican hat wavelet kernel and make plots.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
from gce_utils import get_is_HPC
from astropy.convolution import convolve, RickerWavelet2DKernel
import colorcet as cc
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
nside = 128
npix = hp.nside2npix(nside)
outer_ring = 25
r = outer_ring - 10  # remove the edge to avoid boundary effects
inner_band = 0
n_pixels_c = 100

# Cartesian mapping
proj = hp.projector.CartesianProj(lonra=[-r, r], latra=[-r, r], coord='G', xsize=n_pixels_c, ysize=n_pixels_c)

# Define the map
for sim, tag in zip([I_1, I_2], ['NS_asymmetry', 'NS_shuffled']):
    # Map to Cartesian grid
    # Bartels et al. use 0.1* = 6' (arcmin)
    # at nside = 128: hp pixel size is 27.5'
    # with 100 pixels (per dim.) for -15 to 15: Cart. pixel resolution ~ 0.3*
    # Bartels et al. use sigma = 0.4* for Mexican hat
    # let's use sigma = 1* here for illustration: -> sigma ~ 3.333 pixels

    sim_c = proj.projmap(sim, vec2pix_func=lambda x, y, z: hp.vec2pix(nside, x, y, z, nest=False))
    # plt.imshow(sim_c, origin="lower")

    ricker_kernel = RickerWavelet2DKernel(3.333)

    sim_c_conv = convolve(sim_c, ricker_kernel, normalize_kernel=False, boundary="extend")
    sim_c_conv_2 = convolve(sim_c, ricker_kernel.array ** 2, normalize_kernel=False, boundary="extend")
    eps = 1e-16
    SNR = sim_c_conv / np.sqrt(sim_c_conv_2 + eps)  # see Bartels et al.

    # plt.imshow(ricker_kernel, origin="lower")
    # plt.imshow(sim_c_conv, origin="lower")
    # plt.imshow(sim_c_conv_2, origin="lower")
    fig_SNR, ax_SNR = plt.subplots(1, 1, figsize=(6.4, 4.8))
    fd = {"size": 28}
    fd_label = {"size": 16}
    cmap = cc.cm.CET_D1A
    im = ax_SNR.imshow(SNR, origin="lower", vmin=-10, vmax=10, cmap=cmap)  # , vmin=-20, vmax=20)
    cb = fig_SNR.colorbar(im)
    title = r"$\mathcal{I}_1$" if tag == "NS_asymmetry" else r"$\mathcal{I}_2$"
    ax_SNR.set_title(title, fontdict=fd)
    tick_map = lambda x: -r + x / n_pixels_c * 2 * r
    ticks = np.linspace(0, n_pixels_c, 7)
    ticklabels = np.round(tick_map(ticks)).astype(int)
    ticklabels = [r"$" + str(t) + "$" for t in ticklabels]
    eps_last_tick = 0.001
    ticks[-1] -= eps_last_tick
    ax_SNR.set_xticks(ticks)
    ax_SNR.set_yticks(ticks)
    ax_SNR.set_xticklabels(ticklabels, size=12)
    ax_SNR.set_yticklabels(ticklabels, size=12)
    ax_SNR.set_xlim([0, n_pixels_c - eps_last_tick])
    ax_SNR.set_ylim([0, n_pixels_c - eps_last_tick])
    ax_SNR.set_aspect("equal", anchor="C")
    ax_SNR.set_xlabel(r"Longitude $l$ [$\degree$]", fontdict=fd_label)
    ax_SNR.set_ylabel(r"Latitude $b$ [$\degree$]", fontdict=fd_label)
    cb.set_label("SNR", fontdict=fd_label)
    plt.tight_layout()
    # fig_SNR.savefig("SNR_Wavelet_" + tag + ".pdf")

# Plot kernel
fig_k, ax_k = plt.subplots(1, 1, figsize=(8, 8))
ax_k.imshow(ricker_kernel, origin="lower", cmap=cc.cm.CET_D1A, vmin=-0.0026, vmax=0.0026)
ticks = np.linspace(0, ricker_kernel.shape[0], 7)
ticklabels = np.round(tick_map(ticks + n_pixels_c // 2), 4)
ticklabels = [r"$" + str(t) + "$" for t in ticklabels]
ax_k.set_xticks(ticks)
ax_k.set_yticks(ticks)
ax_k.set_xticklabels(ticklabels, size=12)
ax_k.set_yticklabels(ticklabels, size=12)
plt.tight_layout()
# fig_k.savefig("mexican_hat.pdf")
