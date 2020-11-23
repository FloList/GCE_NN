"""
Iterate over all the training maps and print / save stats
"""
import numpy as np
import pickle
import os

# Running on supercomputer?
if "flo" in os.path.expanduser("~"):
    HPC = False
else:
    HPC = True

if HPC:
    folder_name = "/scratch/u95/fl9575/GCE_v2/data/GCE_maps_toy_hist/gce_12_PS/"
else:
    folder_name = "/home/flo/PycharmProjects/GCE/data/GCE_maps_toy_hist/gce_12_PS/"

# Initialise min. and max. flux
max_flux = 0.0
min_flux = np.infty
max_counts_per_PS = []
above_count_frac = []
below_count_frac = []
exceeds_count_frac_per_PS_mean = []

# Define values to compare to
x_max = 2500  # upper bound for total counts
x_min = 1000  # lower bound for total counts
x_per_PS_max = 60  # upper bound for counts per PS
print_out = True  # print stats for every file?
save_out = False  # save output

# Iterate over the data files
for file in os.listdir(folder_name):
    if "settings" in file:
        continue

    # Get data
    data_file = open(os.path.join(folder_name, file), 'rb')
    data_dict = pickle.load(data_file)
    data_file.close()
    if len(data_dict["data"].shape) == 2:
        this_data = data_dict["data"]
    elif len(data_dict["data"].shape) == 3:  # if second channel is saved with maps without PSF
        this_data = data_dict["data"][:, :, 0]

    # Min. / mean / max. counts
    if print_out:
        print(file, "min. / mean / max. counts:",
              this_data.sum(1).min(), "/", this_data.sum(1).mean(), "/", this_data.sum(1).max())

    # Flux array used for generating the PS: keep track of min./max.
    if "flux_arr" in data_dict.keys():
        fluxes = np.asarray([np.sum(data_dict["flux_arr"][i]) for i in range(len(data_dict["flux_arr"]))])
        if print_out:
            print("   min / mean / max flux: %2.2g / %2.2g / %2.2g" % (fluxes.min(), fluxes.mean(), fluxes.max()))
        min_flux = min(min_flux, fluxes.min())
        max_flux = max(max_flux, fluxes.max())

    # Compute fraction of maps above / below the selected values
    below_count_frac.append(np.sum((this_data.sum(1) < x_min)) / data_dict["data"].shape[0])
    above_count_frac.append(np.sum((this_data.sum(1) > x_max)) / data_dict["data"].shape[0])
    if print_out:
        print(np.round(100 * below_count_frac[-1], 4), "% of the maps have less than", x_min, "counts.")
        print(np.round(100 * above_count_frac[-1], 4), "% of the maps have more than", x_max, "counts.")

    # Check how many maps have point sources with more than x_per_PS_max counts
    if "n_phot" in data_dict.keys():
        max_counts_per_PS.extend([np.max(data_dict["n_phot"][i]) for i in range(len(data_dict["n_phot"])) if len(data_dict["n_phot"][i]) > 0])
        exceeds_count_frac_per_PS_mean.append(np.asarray([np.any(np.asarray(data_dict["n_phot"][i]) > x_per_PS_max) \
                                                          for i in range(len(data_dict["n_phot"])) if len(data_dict["n_phot"][i]) > 0]).mean())
        if print_out:
            print(np.round(100 * exceeds_count_frac_per_PS_mean[-1], 4), "% of the maps have PSs with more than", x_per_PS_max, "counts.")

    if print_out:
        print("+++++++++++++++++++++++++++++++++++++")

# Save
if save_out:
    np.savez("Training_maps_stats.npz", max_flux=max_flux, min_flux=min_flux, max_counts_per_PS=max_counts_per_PS,
             above_count_frac=above_count_frac, below_count_frac=below_count_frac,
             exceeds_count_frac_per_PS_mean=exceeds_count_frac_per_PS_mean)
