"""
This script combines data per model as generated with generate_data_for_SCD_test.py to a combined map and calculates the
flux fraction. Moreover, masks can be applied and the ROI can be reduced.
This script needs to be run twice:
1. STEP = 1: saves the filenames for training and test data
2. STEP = 2: combines the template maps
The reason for this is that the filenames can quickly be written on a supercomputer without the need to submit a proper job (STEP = 1). If this worked smoothly, one can set STEP = 2 and submit a jobscript that combines the maps.
"""
import numpy as np
import healpy as hp
import random
import os
import pickle
import time
import sys
from gce_utils import *
from NPTFit import create_mask as cm  # Module for creating masks
######################################################
# Running on supercomputer?
if "flo" in os.path.expanduser("~"):
    HPC = False
else:
    HPC = True
######################################################
STEP = 1  # first step: 1: save filenames to combine, 2: generate maps
######################################################
M = 1       # 0: no masking
            # 1: confine to ROI
            # 2: also mask 3FGL sources
only_hemisphere = None  # if None: both hemispheres, if "N"/"S": only northern / southern hemisphere
#####################################
gce_dm_on = False
which_SCD = "bright"  # "dim", "def-dim", "default", "def-bright", "bright"
#####################################
# on supercomputer: run this script with arguments: start_train, end_train, start_test, end_test
# (see file submit_combine_data_single.pbs)
if HPC:
    JOB_ID, start_train, end_train, start_test, end_test = 0, None, None, None, None
    try:
        JOB_ID, start_train, end_train, start_test, end_test = sys.argv[1:6]
    except IndexError:
        pass
else:
    JOB_ID = 0
    start_train = end_train = start_test = end_test = None
print("JOB ID is", JOB_ID, ".\n")

######################################
start_time = time.time()

# Settings
fermi_folder = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data' if HPC else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'
input_path = '/scratch/u95/fl9575/GCE_v2/data/GCE_maps_SCD_const_exp' if HPC else '/home/flo/PycharmProjects/GCE/data/GCE_maps_SCD_const_exp'
name = "GCE_and_background_SCD_test_" + which_SCD + "_dm_" + str(gce_dm_on)
filename_out = "GCE_maps"
output_path = '/scratch/u95/fl9575/GCE_v2/data/' + name if HPC else '/home/flo/PycharmProjects/GCE/data/' + name
training_name = "Train"
testing_name = "Test"
outer_ring_min = 25.0  # minimum radius of ROI: each radius will be drawn from a uniform distribution
outer_ring_max = 25.0  # maximum radius of ROI
shuffle_random = True
nside = 128
npix = hp.nside2npix(nside)
allow_different_names = True  # if False: give an error message if the file names for the models are not identical
extend_to_length = 1  # if not None: reuse files in order to generate "extend_to_length" random combinations of
                       # the maps from the individual models (only for training data)
N_test = 0  # number of samples for the test set (have NO contribution in common with training set)

all_models_P = ["iso", "dif", "bub", "gce", "gce_12", "dif_O_pibs", "dif_O_ic", "gce_12_N", "gce_12_S"]
all_models_NP = ["iso_PS", "disk_PS", "thin_disk_PS", "thick_disk_PS", "gce_PS", "gce_12_PS", "bub_PS"]

#############
# CHOOSE MODELS TO COMBINE
T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
T_NP = ["gce_12_PS"]
#############
# Create the output folder (if it doesn't exist yet)
mkdir_p(output_path)

if STEP == 1:
    train_files_dict = dict()
    test_files_dict = dict()
    settings_dict = dict()

    for temp in T_P + T_NP:
        # Get content of folder
        if temp == "gce_12_PS":
            folder = os.path.join(input_path, temp, which_SCD)
        elif temp == "gce_12" and gce_dm_on:
            folder = os.path.join(input_path, temp, "gce_dm_on")
        elif temp == "gce_12" and not gce_dm_on:
            folder = os.path.join(input_path, temp, "gce_dm_off")
        else:
            folder = os.path.join(input_path, temp)
        all_files = os.listdir(folder)
        all_files = np.asarray([file for file in all_files if "EXCLUDE" not in file])  # Don't include files containing "EXCLUDE"
        # Load settings dict
        settings_ind = np.argwhere(["settings" in file for file in all_files])[0][0]
        settings_file = open(os.path.join(folder, all_files[settings_ind]), 'rb')
        settings_dict[temp] = pickle.load(settings_file)
        settings_file.close()
        # Delete settings file from array
        all_files = np.delete(all_files, settings_ind)
        test_files = all_files[:N_test]
        train_files = all_files[N_test:]
        train_length = len(train_files)
        assert train_length >= 1, "N_test is too large for template " + temp + "! No training samples left..."

        # if requested: tile training filenames to get a total length of "extend_to_length"
        if extend_to_length is not None:
            train_files = np.tile(train_files, int(np.ceil(extend_to_length / train_length)))[:extend_to_length]

        # if requested: shuffle
        if shuffle_random:
            random_vec = np.random.choice(np.arange(len(train_files)), len(train_files), replace=False)
            train_files = train_files[random_vec]

        # else: sort
        else:
            sort_vec = np.argsort(train_files)
            train_files = train_files[sort_vec]

        train_files_dict[temp] = train_files
        test_files_dict[temp] = test_files

    # Save the file with the filename dictionary
    with open(os.path.join(output_path, "filenames_combined_train.pickle"), 'wb') as f1:
        pickle.dump(train_files_dict, f1)
        print("Filenames for training saved.")
    with open(os.path.join(output_path, "filenames_combined_test.pickle"), 'wb') as f2:
        pickle.dump(test_files_dict, f2)
        print("Filenames for testing saved.")
    # Save the file with the settings dictionary
    with open(os.path.join(output_path, "settings_combined.pickle"), 'wb') as f3:
        pickle.dump(settings_dict, f3)
        print("Settings files saved.")
        print("EXITING.")
    os._exit(0)

else:
    try:
        train_files_dict = pickle.load(open(os.path.join(output_path, "filenames_combined_train.pickle"), "rb"))
        test_files_dict = pickle.load(open(os.path.join(output_path, "filenames_combined_test.pickle"), "rb"))
        settings_dict = pickle.load(open(os.path.join(output_path, "settings_combined.pickle"), "rb"))
    except (EOFError, IOError, FileNotFoundError):
        print("Run this script with STEP=1 first!")
        os._exit(1)

# Get exposure and unmasked pixels
assert train_files_dict.keys() == test_files_dict.keys(), "Training and testing dictionaries have different keys!"
unmasked_pix = np.unique(np.asarray([settings_dict[key]["unmasked_pix"] for key in [*train_files_dict]]), axis=0).squeeze()
exp = np.unique(np.asarray([settings_dict[key]["exp"] for key in [*train_files_dict]]), axis=0).squeeze()
mean_exp = np.mean(exp)
rescale = exp / mean_exp

# Get number of files
n_files_train = len(train_files_dict[[*train_files_dict][0]])
n_files_test = len(test_files_dict[[*test_files_dict][0]])

# Set filenames of output files
if filename_out is None:
    if allow_different_names:
        files_train = train_files_dict[[*train_files_dict][0]]  # take file names of the template
        files_test = test_files_dict[[*test_files_dict][0]]
    else:
        files_train = np.unique(np.asarray([train_files_dict[key] for key in [*train_files_dict]]), axis=0).squeeze()
        files_test = np.unique(np.asarray([test_files_dict[key] for key in [*test_files_dict]]), axis=0).squeeze()
    settings_filename = "settings.pickle"
else:
    no_dig = int(np.ceil(np.log10(n_files_train)))
    files_train = np.asarray([filename_out + "_" + str(i).zfill(no_dig) + "_train.pickle" for i in range(n_files_train)])
    files_test = np.asarray([filename_out + "_" + str(i).zfill(no_dig) + "_test.pickle" for i in range(n_files_test)])
    settings_filename = filename_out + "_settings.pickle"

# Set up the mask
if M == 2:
    pscmask = np.load(os.path.join(fermi_folder, "fermidata_pscmask.npy"))
    total_mask_neg = (1 - (1 - cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=outer_ring_max)) * (1 - pscmask))
elif M == 1:
    total_mask_neg = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=outer_ring_max)
else:
    total_mask_neg = np.zeros(npix)

# Only one hemisphere?
if only_hemisphere is not None:
    total_mask_pos_N, total_mask_pos_S = split_into_N_and_S(1 - total_mask_neg, nside=nside, filename=None)
    total_mask_neg = (1 - total_mask_pos_S) if only_hemisphere == "S" else (1 - total_mask_pos_N)

# Make folders
mkdir_p(os.path.join(output_path, training_name))
mkdir_p(os.path.join(output_path, testing_name))

# Set ranges of files to write
start_train = 0 if start_train is None else int(start_train)
start_test = 0 if start_test is None else int(start_test)
end_train = n_files_train - 1 if end_train is None else int(end_train)
end_test = n_files_test - 1 if end_test is None else int(end_test)

# Do the following for the training and the test samples
for is_testing, files_dict in enumerate([train_files_dict, test_files_dict]):

    # Set training / testing specific settings
    n_files = n_files_test if is_testing else n_files_train
    files = files_test if is_testing else files_train
    outfolder = os.path.join(output_path, testing_name if is_testing else training_name)
    start = start_test if is_testing else start_train
    end = end_test if is_testing else end_train

    # Check correctness
    assert np.all([len(files_dict[key]) == len(files_dict[[*files_dict][0]]) for key in [*files_dict]]), \
        "The number of files for each model is not the same! Consider setting extend_to_length. Aborting..."

    if not allow_different_names and not shuffle_random:
        assert np.all([np.all(files_dict[key] == files_dict[[*files_dict][0]]) for key in [*files_dict]]), \
            "The file names for each model are not the same! Aborting..."

    # Write a combined settings dictionary (need to reorder: temp -> key => key -> temp)
    if not is_testing:
        if int(JOB_ID) == 0:
            settings_dict_comb = dict()
            all_keys = ["T", "T_corr", "priors", "max_NP_sources"]
            for key in all_keys:
                temp_dict = dict()
                for temp in T_P + T_NP:
                    temp_keys = list(settings_dict[temp].keys())
                    if key in temp_keys:
                        temp_dict[temp] = settings_dict[temp][key]
                settings_dict_comb[key] = temp_dict
            settings_dict_comb["exp"] = exp
            settings_dict_comb["rescale"] = rescale

            # Store the new unmasked pixels (may be less due to added mask / smaller ROI)
            settings_dict_comb["unmasked_pix"] = np.argwhere(1 - total_mask_neg).flatten()

            if os.path.isfile(os.path.join(output_path, training_name, settings_filename)):
                print("Settings file exists already...")
            else:
                with open(os.path.join(output_path, training_name, settings_filename), 'wb') as f:
                    pickle.dump(settings_dict_comb, f)
                    print("Combined settings file written.")

    # Print info
    dash = 80 * "="
    print(dash)
    print("NAME:", name)
    print("JOB_ID =", JOB_ID)
    print("=== TESTING DATA ===" if is_testing else "=== TRAINING DATA ===")
    print("Starting to combine the maps {0} ... {1} out of {2} in total.".format(start, end, n_files))
    print("Poissonian models:", T_P)
    print("Non-Poissonian models:", T_NP)
    if M == 0:
        print("No masking.")
    elif M == 1:
        print("Confining to ring of radius", outer_ring_min, "-", outer_ring_max, "degrees.")
    elif M == 2:
        print("Confining to ring of radius", outer_ring_min, "-", outer_ring_max, "degrees and masking 3FGL sources.")
    if extend_to_length is not None and not is_testing:
        print("Maps will be used repeatedly to generate a total of", extend_to_length, "combined maps.")
    if shuffle_random:
        print("Random shuffling of the template maps is ON.")
    else:
        print("Random shuffling of the template maps is OFF.")
    print(dash + "\n")

    for i_file in range(start, end + 1):
        if os.path.isfile(os.path.join(outfolder, files[i_file])):
            print("File", i_file, "exists... continue")
            continue
        total_flux_dict = dict()
        flux_fraction_dict = dict()
        data_dict = dict()
        data_out = dict()
        combined_map = 0

        for i_temp, temp in enumerate(T_P + T_NP):
            if temp == "gce_12_PS":
                data_file = open(os.path.join(input_path, temp, which_SCD, files_dict[temp][i_file]), 'rb')
            elif temp == "gce_12" and gce_dm_on:
                data_file = open(os.path.join(input_path, temp, "gce_dm_on", files_dict[temp][i_file]), 'rb')
            elif temp == "gce_12" and not gce_dm_on:
                data_file = open(os.path.join(input_path, temp, "gce_dm_off", files_dict[temp][i_file]), 'rb')
            else:
                data_file = open(os.path.join(input_path, temp, files_dict[temp][i_file]), 'rb')

            # Load data
            data_dict[temp] = pickle.load(data_file)
            data_file.close()
            # if outer_ring_min != outer_ring_max: shrink ROIs
            if i_temp == 0 and (outer_ring_min != outer_ring_max):
                rad_ROI = np.asarray([random.uniform(outer_ring_min, outer_ring_max) for _ in range(data_dict[temp]["data"].shape[1])])
                this_mask = ((1 - np.expand_dims(total_mask_neg, -1)) * np.asarray([(1 - cm.make_mask_total(nside=nside, mask_ring=True, inner=0, outer=rad)) for rad in rad_ROI]).T)
            elif outer_ring_min == outer_ring_max:
                this_mask = (1 - np.expand_dims(total_mask_neg, -1))
            # Get the full map
            full_map = masked_to_full(data_dict[temp]["data"].T, unmasked_pix, nside=nside).T
            # Confine to ROI and do masking if required
            ROI_map = full_map * this_mask
            # Add to combined map
            combined_map += ROI_map
            # Calculate flux
            flux = ROI_map / np.expand_dims(exp, -1)
            total_flux_dict[temp] = flux.sum(0)

        # Calculate flux fractions
        total_flux = np.asarray([v for k, v in total_flux_dict.items()]).sum(0)
        for temp in T_P + T_NP:
            flux_fraction_dict[temp] = total_flux_dict[temp] / total_flux

        # Write combined info
        info_dict_comb = dict()
        all_keys = ["A", "A_corr", "n", "S", "F"]
        for key in all_keys:
            temp_dict = dict()
            for temp in T_P + T_NP:
                temp_keys = list(data_dict[temp]["info"].keys())
                if key in temp_keys:
                    temp_dict[temp] = data_dict[temp]["info"][key]
            info_dict_comb[key] = temp_dict

        # Store in "data_out" dictionary
        data_out["data"] = combined_map[(1 - total_mask_neg).astype(bool), :]
        data_out["flux_fraction"] = flux_fraction_dict
        data_out["info"] = info_dict_comb

        with open(os.path.join(outfolder, files[i_file]), 'wb') as f:
            pickle.dump(data_out, f)

        # Print some stats
        print("File {0} / {1}:".format(i_file + 1, len(files)))
        print(dash)
        print("Number of simulations: {0}".format(data_out["data"].shape[1]))
        print("Templates:")
        print(list(data_out["flux_fraction"].keys()))
        print("Max. flux fraction for each template:")
        print([np.round(data_out["flux_fraction"][key].max(), 2) for key in data_out["flux_fraction"].keys()])
        print("Min. flux fraction for each template:")
        print([np.round(data_out["flux_fraction"][key].min(), 2) for key in data_out["flux_fraction"].keys()])
        print("Mean flux fraction for each template:")
        print([np.round(data_out["flux_fraction"][key].mean(), 2) for key in data_out["flux_fraction"].keys()])
        print("Median flux fraction for each template:")
        print([np.round(np.median(data_out["flux_fraction"][key]), 2) for key in data_out["flux_fraction"].keys()])
        print("Avg. total number of counts:")
        print(np.round(np.mean(combined_map.sum(0))))
        if outer_ring_min != outer_ring_max:
            print("Radius of ROI between", np.round(rad_ROI.min(), 2), "and", np.round(rad_ROI.max(), 2))
        print(dash + "\n")

        auto_garbage_collect()
