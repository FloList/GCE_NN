"""
This script can be used for data augmentation (mirroring) of maps saved with "generate_data_per_model.py".
NOTE: THIS MAKES ONLY SENSE IF THE EXPOSURE MAP HAS SYMMETRIES AS WELL (for example for uniform exposure)!
"""
import os
import numpy as np
import pickle
from gce_utils import flip_map, masked_to_full
######################################################
# Running on supercomputer?
if "flo" in os.path.expanduser("~"):
    HPC = False
else:
    HPC = True
######################################################
from NPTFit import create_mask as cm  # Module for creating masks
######################################################
# Settings
output_path = '/scratch/u95/fl9575/GCE_v2/data/GCE_maps_toy_hist' if HPC else "/home/flo/PycharmProjects/GCE/data/GCE_maps_toy_hist"

all_models_P = ["iso", "dif", "bub", "gce", "gce_12", "dif_O_pibs", "dif_O_ic"]
all_models_NP = ["iso_PS", "disk_PS", "gce_PS", "gce_12_PS", "bub_PS"]

nside = 128
#############
# CHOOSE MODELS AND SET DATA AUGMENTATION METHODS
T = ["gce_12_PS"]
flip_H = [True]
flip_V = [True]
#############

# Print
print("Starting data augmentation.")
print("WARNING: MAKE SURE THAT BOTH THE TEMPLATES AND THE EXPOSURE MAP HAVE THE SPECIFIED SYMMETRIES!")

# Data augmentation
for i_temp, temp in enumerate(T):
    temp_folder = os.path.join(output_path, temp)
    print("Starting with model", temp)
    all_files = os.listdir(temp_folder)
    settings_ind = np.argwhere(["settings" in file for file in all_files])[0][0]
    settings_file = open(os.path.join(temp_folder, all_files[settings_ind]), 'rb')
    settings_dict = pickle.load(settings_file)
    settings_file.close()
    data_files = [file for file in all_files if "EXCLUDE" not in file and "settings" not in file]

    # Go through all files
    for i_file, file in enumerate(data_files):
        this_file = open(os.path.join(temp_folder, file), 'rb')
        this_data = pickle.load(this_file)
        this_file.close()

        if len(this_data["data"].shape) == 2:  # shape: (n_maps, n_pix)
            this_map = masked_to_full(this_data["data"], settings_dict["unmasked_pix"], nside=nside).T
        elif len(this_data["data"].shape) == 3:  # shape: (n_maps, n_pix, 2) (with PSF, without PSF)
            this_map = np.stack([masked_to_full(this_data["data"][:, :, 0], settings_dict["unmasked_pix"], nside=nside).T,
                                 masked_to_full(this_data["data"][:, :, 1], settings_dict["unmasked_pix"], nside=nside).T], 2)


        # Save horizontally flipped map
        if flip_H[i_temp]:
            if len(this_data["data"].shape) == 2:
                map_H = flip_map(this_map, hor=True)
            elif len(this_data["data"].shape) == 3:
                map_H = np.transpose(np.stack([flip_map(this_map[:, :, 0], hor=True),
                                               flip_map(this_map[:, :, 1], hor=True)], 2), [1, 0, 2])
            with open(os.path.join(temp_folder, file + "_H_flipped.pickle"), 'wb') as f:
                new_dict = this_data.copy()
                new_dict["data"] = map_H[:, settings_dict["unmasked_pix"]]
                assert np.all(new_dict["data"].sum(1) == this_map.sum(0))
                pickle.dump(new_dict, f)

        # Save vertically flipped map
        if flip_V[i_temp]:
            if len(this_data["data"].shape) == 2:
                map_V = flip_map(this_map, vert=True)
            elif len(this_data["data"].shape) == 3:
                map_V = np.transpose(np.stack([flip_map(this_map[:, :, 0], vert=True),
                                               flip_map(this_map[:, :, 1], vert=True)], 2), [1, 0, 2])
            with open(os.path.join(temp_folder, file + "_V_flipped.pickle"), 'wb') as f:
                new_dict = this_data.copy()
                new_dict["data"] = map_V[:, settings_dict["unmasked_pix"]]
                assert np.all(new_dict["data"].sum(1) == this_map.sum(0))
                pickle.dump(new_dict, f)

        # Save doubly flipped map
        if flip_H[i_temp] and flip_V[i_temp]:
            if len(this_data["data"].shape) == 2:
                map_HV = flip_map(this_map, hor=True, vert=True)
            elif len(this_data["data"].shape) == 3:
                map_HV = np.transpose(np.stack([flip_map(this_map[:, :, 0], hor=True, vert=True),
                                                flip_map(this_map[:, :, 1], hor=True, vert=True)], 2), [1, 0, 2])
            with open(os.path.join(temp_folder, file + "_HV_flipped.pickle"), 'wb') as f:
                new_dict = this_data.copy()
                new_dict["data"] = map_HV[:, settings_dict["unmasked_pix"]]
                assert np.all(new_dict["data"].sum(1) == this_map.sum(0))
                pickle.dump(new_dict, f)

print("Done!")
