import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits
from GCE.data_utils import make_ring_mask, make_long_mask, make_lat_mask

# Folders and file names
folder = "/home/flo/Documents/Projects/GCE_hist/Comparison_project/Data"
hp_model_silvia_128_name = folder + "/DiffuseModelE_2GeV_5GeV_P8R3_ULTRACLEANVETO_V2_evtype3_40x40deg_0p5deg_hp.fits"  # Binned to Healpix nside = 128 by Silvia
hp_model_silvia_40x40_name = folder + "/DiffuseModelE_2GeV_5GeV_P8R3_ULTRACLEANVETO_V2_evtype3_40x40deg_0p5deg.fits"  # Original: 40x40 deg
psf_file = os.path.join(folder, "Numpy", "psf_10_bins_300_angles.npz")  # PSF file
out_folder = os.path.join(folder, "Numpy")

# First: load exposure data
exp_file_256 = os.path.join(folder, "Numpy", "fermi_data_256", "exp_10_bins.npy")
exp_256 = np.load(exp_file_256)
mask_30_deg_256 = make_ring_mask(inner=0.0, outer=30.0, ring_b=0.0, ring_l=0.0, nside=256)
mean_exp_in_roi_256 = np.mean(exp_256[:, ~mask_30_deg_256], axis=1)

exp_128 = hp.ud_grade(exp_256, power=-2, nside_out=128)
mask_30_deg_128 = make_ring_mask(inner=0.0, outer=30.0, ring_b=0.0, ring_l=0.0, nside=128)
mean_exp_in_roi_128 = np.mean(exp_128[:, ~mask_30_deg_128], axis=1)

# Load the diffuse model
# Open the FITS files
with fits.open(hp_model_silvia_128_name) as hdul:
    hdul.info()
    hp_model_silvia_128 = np.asarray(hdul[0].data)

# with fits.open(hp_model_silvia_40x40_name) as hdul:
#     hdul.info()
#     hp_model_silvia_40x40 = np.asarray(hdul[0].data)

# Also upscale to nside = 256
hp_model_silvia_256 = hp.ud_grade(hp_model_silvia_128, power=-2, nside_out=256)

# Plot the models
hp.mollview(np.log10(hp_model_silvia_128), title="Silvia's model 128")
hp.mollview(np.log10(hp_model_silvia_256), title="Silvia's model 256")

# Also plot the models only in the ROI
# Get a mask for the region of interest: 20x20 deg -> this is the right mask!
mask_128 = make_long_mask(-20, 20, nside=128) | make_lat_mask(-20, 20, nside=128)
unmasked_128 = ~mask_128
mask_128 = 1.0 * mask_128
mask_128[unmasked_128] = hp_model_silvia_128[unmasked_128]
mask_128[~unmasked_128] = np.nan
hp.mollview(np.log10(mask_128), title="Silvia's model in ROI 128")

mask_256 = make_long_mask(-20, 20, nside=256) | make_lat_mask(-20, 20, nside=256)
unmasked_256 = ~mask_256
mask_256 = 1.0 * mask_256
mask_256[unmasked_256] = hp_model_silvia_256[unmasked_256]
mask_256[~unmasked_256] = np.nan
hp.mollview(np.log10(mask_256), title="Silvia's model in ROI 256")

# Now: exposure correction!
hp_model_silvia_128_corr = np.array([hp_model_silvia_128 * exp_128[i] / mean_exp_in_roi_128[i] for i in range(10)])
hp_model_silvia_256_corr = np.array([hp_model_silvia_256 * exp_256[i] / mean_exp_in_roi_256[i] for i in range(10)])

# Plot for two energies
hp.mollview(np.log10(hp_model_silvia_128_corr[0]), title="Silvia's model exp-corr @ 2 GeV 128")
hp.mollview(np.log10(hp_model_silvia_256_corr[0]), title="Silvia's model exp-corr @ 2 GeV 256")

# Now: smooth the diffuse model with the PSF
# Load the PSF
psf_data = np.load(psf_file)
psf_energies = psf_data["energies"]
psf_theta_rad = psf_data["theta_rad"]
psf_values = psf_data["psf"]

# Compute the beam window function
lmax_128 = 3 * 128 - 1
lmax_256 = 3 * 256 - 1
beam_psf_128 = [hp.sphtfunc.beam2bl(psf, psf_theta_rad, lmax_128) for psf in psf_values]
beam_psf_128 = [np.heaviside(psf / psf.max(), 0.0) * psf / psf.max() for psf in beam_psf_128]
beam_psf_256 = [hp.sphtfunc.beam2bl(psf, psf_theta_rad, lmax_256) for psf in psf_values]
beam_psf_256 = [np.heaviside(psf / psf.max(), 0.0) * psf / psf.max() for psf in beam_psf_256]

# Smooth the diffuse model with the PSF
nside_multiplier = 4
get_smoothed_map_128 = lambda x, i_bin, nside_loc: hp.ud_grade(hp.smoothing(hp.ud_grade(x, nside_multiplier * nside_loc,
                                                                                        power=-2),
                                                                            beam_window=beam_psf_128[i_bin],
                                                                            lmax=lmax_128), nside_loc, power=-2)

get_smoothed_map_256 = lambda x, i_bin, nside_loc: hp.ud_grade(hp.smoothing(hp.ud_grade(x, nside_multiplier * nside_loc,
                                                                                        power=-2),
                                                                            beam_window=beam_psf_256[i_bin],
                                                                            lmax=lmax_256), nside_loc, power=-2)


smoothed_dif_maps_corr_128 = np.array([get_smoothed_map_128(hp_model_silvia_128_corr[i], i, 128) for i in range(10)])
smoothed_dif_maps_corr_256 = np.array([get_smoothed_map_256(hp_model_silvia_256_corr[i], i, 256) for i in range(10)])

# Make a plot for two energies
m, M = -8, -3.8
hp.mollview(np.log10(smoothed_dif_maps_corr_128[0]), min=m, max=M, title="Silvia's model smoothed & exp-corr @ 2 GeV 128")
hp.mollview(np.log10(smoothed_dif_maps_corr_256[0]), min=m, max=M, title="Silvia's model smoothed & exp-corr @ 2 GeV 256")
hp.mollview(np.log10(smoothed_dif_maps_corr_128[-1]), min=m, max=M, title="Silvia's model smoothed & exp-corr @ 5 GeV 128")
hp.mollview(np.log10(smoothed_dif_maps_corr_256[-1]), min=m, max=M, title="Silvia's model smoothed & exp-corr @ 5 GeV 256")

# Also store the isotropic PS model
iso_model_128 = exp_128 / mean_exp_in_roi_128[:, None]
iso_model_256 = exp_256 / mean_exp_in_roi_256[:, None]

# Plot for two energy bins
hp.mollview(np.log10(iso_model_128[0]), title="Isotropic PS exp-corr @ 2 GeV 128")
hp.mollview(np.log10(iso_model_256[0]), title="Isotropic PS exp-corr @ 2 GeV 256")
hp.mollview(np.log10(iso_model_128[-1]), title="Isotropic PS exp-corr @ 5 GeV 128")
hp.mollview(np.log10(iso_model_256[-1]), title="Isotropic PS exp-corr @ 5 GeV 256")
# Do NOT convolve with the PSF, as this is a PS template!

# Save everything
np.save(os.path.join(out_folder, "fermi_data_128", "template_diffuse_silvia_smooth.npy"), smoothed_dif_maps_corr_128)
np.save(os.path.join(out_folder, "fermi_data_256", "template_diffuse_silvia_smooth.npy"), smoothed_dif_maps_corr_256)
np.save(os.path.join(out_folder, "fermi_data_128", "template_iso.npy"), iso_model_128)
np.save(os.path.join(out_folder, "fermi_data_256", "template_iso.npy"), iso_model_256)


# # PSF testing
# import healpy as hp
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def top_hat(b, radius):
#     return np.where(abs(b) <= radius, 1, 0)
#
#
# nside = 128
# npix = hp.nside2npix(nside)
#
# # create a empy map
# tst_map = np.zeros(npix)
#
# # put a source in the middle of the map with value = 100
# pix = hp.ang2pix(nside, np.pi / 2, 0)
# tst_map[pix] = 1000000
#
# # Compute the window function in the harmonic spherical space which will smooth the map.
# b = np.linspace(0, np.pi, 10000)
# bw = top_hat(b, np.radians(125))  # top_hat function of radius
# beam = hp.sphtfunc.beam2bl(bw, b, nside * 3)
#
# # Smooth map
# tst_map_smoothed = hp.smoothing(tst_map, beam_window=beam)
#
# hp.mollview(tst_map_smoothed)
# plt.show()
