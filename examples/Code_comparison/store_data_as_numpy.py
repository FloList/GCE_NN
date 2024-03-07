import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits                                             

# Load FITS files
folder = "/home/flo/Documents/Projects/GCE_hist/Comparison_project/Data"
subfolder = "diffE_isops_dNdS1"
outfolder = "Numpy"

# Exposure file
exp_file = os.path.join(folder, subfolder, "expcube2.fits")

# Open the FITS file
with fits.open(exp_file) as hdul:
    hdul.info()
    exp = np.asarray(hdul[1].data)
    energies = np.asarray(hdul[2].data)["Energy"]

exp_E_x_pix = np.asarray([exp[f"ENERGY{i}"] for i in range(1, 11)])

# Store
# np.save(os.path.join(folder, outfolder, "exp_10_bins.npy"), exp_E_x_pix)
# np.save(os.path.join(folder, outfolder, "energies.npy"), energies)

# Plot the exposure
exp_mean_E = np.mean(exp_E_x_pix, axis=0)
hp.mollview(exp_mean_E, title="Exposure")

# PSF file
psf_file = os.path.join(folder, subfolder, "psffid.fits")

# Open the FITS file
with fits.open(psf_file) as hdul:
    hdul.info()
    psf_data = np.asarray(hdul[1].data)
    psf_theta_deg = np.asarray(hdul[2].data)["Theta"]
    psf_theta_rad = np.radians(psf_theta_deg)

print("Energy shape: ", psf_data["Energy"].shape)
print("Exposure shape: ", psf_data["Exposure"].shape)
print("PSF shape: ", psf_data["Psf"].shape)
print("Theta shape: ", psf_theta_rad.shape)

# Make a plot of the PSF as a function of theta, for all energies
colors = plt.cm.viridis(np.linspace(0, 1, 10))
fig, ax = plt.subplots(1, 1)
for i in range(10):
    ax.plot(psf_theta_rad, psf_data["Psf"][i], label=f"{energies[i]:.2f} MeV", color=colors[i])
ax.set_yscale("log")
ax.set_xlabel("Theta (rad)")
ax.set_ylabel("PSF")

# Compare with King function (for 2 GeV)
from GCE.data_utils import fermi_psf, get_fermi_pdf_sampler
psf_king = fermi_psf(psf_theta_rad)
normalization = psf_data["Psf"][0][0] / psf_king[0]
ax.plot(psf_theta_rad, normalization * psf_king, label="King function for 2 GeV", color="r", linestyle="--", lw=2)
ax.legend()

# Store the PSF
assert np.allclose(energies, psf_data["Energy"])
# np.savez(os.path.join(folder, outfolder, "psf_10_bins_300_angles.npz"),
#          psf=psf_data["Psf"], energies=energies, theta_rad=psf_theta_rad)

# Now, load the count maps
subfolders = ["diffE_isops_dNdS1", "diffE_isops_dNdS2", "diffE_isops_dNdS3"]
for subfolder in subfolders:
    print(f"Processing {subfolder}")
    count_map_file = os.path.join(folder, subfolder, "gtbin_out.fits")

    # Open the FITS file
    with fits.open(count_map_file) as hdul:
        hdul.info()
        count_data = np.asarray(hdul[1].data)
        energies_data = np.asarray(hdul[2].data)

    count_map = np.asarray([count_data[f"CHANNEL{i}"] for i in range(1, 11)])
    energy_min = energies_data["E_MIN"]
    energy_max = energies_data["E_MAX"]
    assert np.allclose(energy_min[1:], energy_max[:-1])
    energy_boundaries = np.concatenate((energy_min[:1], energy_max))
    # np.save(os.path.join(folder, outfolder, f"count_map_{subfolder[-1]}_256.npy"), count_map)
    hp.mollview(count_map[0], title="Count map at 2 GeV")
    hp.mollview(count_map[-1], title="Count map at 5 GeV")
    hp.mollview(count_map.sum(0), title="Count map summed over all energies")
    # Also print the total number of counts, and plot the spectrum
    total_counts = np.sum(count_map, axis=(0, 1))
    print("Total counts: ", total_counts)
    plt.figure()
    plt.plot(count_map.sum(1))
