"""
This combines data per model to a combined map and calculates the flux fractions.
Here, two maps are randomly added for a specified template (simulating GCE DM + PS / 2 PS populations in a single map).
Moreover, histograms are generated from the n_phot lists.
This script needs to be run TWICE:
1. SAVE_FILENAMES = True: saves the filenames for training and test data
2. SAVE_FILENAMES = False: combines the template maps
NOTE: in contrast to other combining functions, NO MASKS can be applied here and the ROI can NOT be changed!
This is because neither the list n_phot nor dN/dF contain any local information, for which reason masking parts of
the maps in this function would destroy the map <-> histogram correspondence.
"""
import numpy as np
import healpy as hp
import random
import os
import pickle
import time
import sys
from gce_utils import *
######################################################
# Running on supercomputer?
HPC = get_is_HPC()
######################################################
SAVE_FILENAMES = False  # first step: True: save filenames to combine, second step: False: generate maps
######################################################
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
ADD_template = "gce_12_PS"  # NOTE: twice as many files need to be available for this template!
nside = 256
name_in = 'GCE_maps_573w'
input_path = '/scratch/u95/fl9575/GCE_v2/data/' if HPC else '/home/flo/PycharmProjects/GCE/data/'
input_path = os.path.join(input_path, name_in)
input_path += "_" + str(nside)
name = "Fermi_example_add_two_256"  # folder name for combined data
filename_out = "Count_maps"  # filename basis for combined maps
output_path = '/scratch/u95/fl9575/GCE_v2/data/' + name if HPC else '/home/flo/PycharmProjects/GCE/data/' + name
training_name = "Train"
testing_name = "Test"
shuffle_random = True
npix = hp.nside2npix(nside)
N_test = 1000  # number of files for the test set (have NO contribution in common with training set)

# Histogram settings
do_dNdF = True  # save histograms of dNdFs?
do_counts_per_PS = False  # save histograms of counts per PS?
do_counts_per_pix = False  # save histograms of counts per pixel before the PSF (as saved in the 2nd data channel)?
# Define the histogram bins
bins_dNdF = np.asarray([-np.infty] + list(np.logspace(-12.5, -7.0, 21)) + [np.infty])
power_of_F_dNdF = 1  # power of F to multiply dN/dF with
bins_counts_per_PS = bins_counts_per_pix = None
# Hist template name
hist_templates = ["gce_12_PS", "thin_disk_PS"]

#############
# CHOOSE MODELS TO COMBINE
T_P = ["dif_O_pibs", "dif_O_ic", "iso", "bub"]
T_NP = ["gce_12_PS", "thin_disk_PS"]
#############
# Create the output folder (if it doesn't exist yet)
mkdir_p(output_path)
dash = 80 * "="
do_any_hist = (do_dNdF or do_counts_per_PS or do_counts_per_pix) and (len(hist_templates) > 0)

files_dict_no_add_two = np.setdiff1d(T_P + T_NP, ADD_template)

if SAVE_FILENAMES:
    train_files_dict = dict()
    test_files_dict = dict()
    settings_dict = dict()

    for temp in T_P + T_NP:
        # Get content of folder
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
        if temp == ADD_template:
            test_files = all_files[:2 * N_test]
            train_files = all_files[2 * N_test:]
        else:
            test_files = all_files[:N_test]
            train_files = all_files[N_test:]
        train_length = len(train_files)
        assert train_length >= 1 + (temp == ADD_template), \
            "N_test is too large for template " + temp + "! No training samples left..."

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
        print("Run this script with SAVE_FILENAMES=True first!")
        os._exit(1)

# Get exposure and unmasked pixels
assert train_files_dict.keys() == test_files_dict.keys(), "Training and testing dictionaries have different keys!"
unmasked_pix = np.unique(np.asarray([settings_dict[key]["unmasked_pix"] for key in [*train_files_dict]]), axis=0).squeeze()
exp = np.unique(np.asarray([settings_dict[key]["exp"] for key in [*train_files_dict]]), axis=0).squeeze()
mean_exp = np.mean(exp)
rescale = exp / mean_exp

# Get number of files
n_files_train = len(train_files_dict[files_dict_no_add_two[0]])
n_files_test = len(test_files_dict[files_dict_no_add_two[0]])

# Set filenames of output files
if filename_out is None:
    files_train = train_files_dict[files_dict_no_add_two[0]]  # take file names of the template
    files_test = test_files_dict[files_dict_no_add_two[0]]
    settings_filename = "settings.pickle"
else:
    no_dig = int(np.ceil(np.log10(n_files_train)))
    files_train = np.asarray([filename_out + "_" + str(i).zfill(no_dig) + "_train.pickle" for i in range(n_files_train)])
    files_test = np.asarray([filename_out + "_" + str(i).zfill(no_dig) + "_test.pickle" for i in range(n_files_test)])
    settings_filename = filename_out + "_settings.pickle"

# Make folders
mkdir_p(os.path.join(output_path, training_name))
mkdir_p(os.path.join(output_path, testing_name))

# Set ranges of files to write
start_train = 0 if start_train is None else int(start_train)
start_test = 0 if start_test is None else int(start_test)
end_train = n_files_train - 1 if end_train is None else int(end_train)
end_test = n_files_test - 1 if end_test is None else int(end_test)

# Back up this file
if int(JOB_ID) == 0:
    backup_folder = os.path.join(output_path, "src_backup")
    mkdir_p(backup_folder)
    backup_one_file(__file__, backup_folder)

# Do the following for the training and the test samples
for is_testing, files_dict in enumerate([train_files_dict, test_files_dict]):

    # Set training / testing specific settings
    n_files = n_files_test if is_testing else n_files_train
    files = files_test if is_testing else files_train
    outfolder = os.path.join(output_path, testing_name if is_testing else training_name)
    start = start_test if is_testing else start_train
    end = end_test if is_testing else end_train

    # Check correctness
    assert np.all([len(files_dict[key]) == len(files_dict[files_dict_no_add_two[0]]) for key in files_dict_no_add_two]), \
        "The number of files for each model is not the same! Consider setting extend_to_length. Aborting..."

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
            unmasked_pix_all = np.asarray([settings_dict[temp]["unmasked_pix"] for temp in T_P + T_NP])
            unmasked_pix_unique = np.unique(unmasked_pix_all, axis=0)
            if unmasked_pix_unique.shape[0] > 1:
                raise ValueError("Unmasked pixels for the different template maps are not identical!")
            settings_dict_comb["unmasked_pix"] = unmasked_pix_unique.flatten()
            exp_unmasked_pix = exp[settings_dict_comb["unmasked_pix"]]
            format_all = np.asarray([settings_dict[temp]["format"] for temp in T_P + T_NP])
            format_unique = np.unique(format_all)
            if len(format_unique) > 1:
                raise ValueError("Different formats (RING / NEST) for the different templates are not supported!")
            settings_dict_comb["format"] = format_unique[0]

            # Histogram-specific settings
            if do_any_hist:
                settings_dict_comb["hist_bins"] = dict()
                if do_dNdF:
                    settings_dict_comb["hist_bins"]["dNdF"] = bins_dNdF
                    settings_dict_comb["hist_bins"]["power_of_F_dNdF"] = power_of_F_dNdF
                if do_counts_per_PS:
                    settings_dict_comb["hist_bins"]["counts_per_PS"] = bins_counts_per_PS
                if do_counts_per_pix:
                    settings_dict_comb["hist_bins"]["counts_per_pix"] = bins_counts_per_pix

            if os.path.isfile(os.path.join(output_path, training_name, settings_filename)):
                print("Settings file exists already...")
            else:
                with open(os.path.join(output_path, training_name, settings_filename), 'wb') as f:
                    pickle.dump(settings_dict_comb, f)
                    print("Combined settings file written.")

    # in any case: need to get exp_unmasked_pix
    unmasked_pix_all = np.asarray([settings_dict[temp]["unmasked_pix"] for temp in T_P + T_NP])
    unmasked_pix_unique = np.unique(unmasked_pix_all, axis=0)
    if unmasked_pix_unique.shape[0] > 1:
        raise ValueError("Unmasked pixels for the different template maps are not identical!")
    exp_unmasked_pix = exp[unmasked_pix_unique.flatten()]

    # Print info
    print(dash)
    print("NAME:", name)
    print("JOB_ID =", JOB_ID)
    print("=== TESTING DATA ===" if is_testing else "=== TRAINING DATA ===")
    print("Starting to combine the maps {0} ... {1} out of {2} in total.".format(start, end, n_files))
    print("Poissonian models:", T_P)
    print("Non-Poissonian models:", T_NP)
    if shuffle_random:
        print("Random shuffling of the template maps is ON.")
    else:
        print("Random shuffling of the template maps is OFF.")
    print("EACH MAP WILL CONTAIN 2 TEMPLATE MAPS OF TEMPLATE", ADD_template + "!")

    print(dash + "\n")

    for i_file in range(start, end + 1):
        if os.path.isfile(os.path.join(outfolder, files[i_file])):
            print("File", i_file, "exists... continue")
            continue
        total_flux_dict = dict()
        flux_fraction_dict = dict()
        data_dict = dict()
        data_dict_0, data_dict_1 = dict(), dict()
        data_out = dict()
        combined_map = 0

        for i_temp, temp in enumerate(T_P + T_NP):
            total_flux_dict[temp] = 0.0
            is_add_two = temp == ADD_template
            repeat_range = [0, 1] if is_add_two else [0]
            for rep in repeat_range:
                if is_add_two:
                    this_file = files_dict[temp][2 * i_file + rep]
                else:
                    this_file = files_dict[temp][i_file]
                data_file = open(os.path.join(input_path, temp, this_file), 'rb')
                temp_data = pickle.load(data_file)
                temp_map = temp_data["data"]  # n_maps_per_chunk x n_unmasked_pixs
                if is_add_two:
                    if rep == 0:
                        data_dict_0[temp] = temp_data
                    else:
                        data_dict_1[temp] = temp_data
                    data_file.close()
                else:
                    # Load data
                    data_dict[temp] = temp_data
                # If PS map with second channel for data without PSF
                if len(temp_map.shape) == 3:
                    temp_map = temp_map[:, :, 0]
                # Add to combined map
                combined_map += temp_map
                # Calculate flux
                flux = temp_map / np.expand_dims(exp_unmasked_pix, 0)
                # Total flux of template: sum over pixels
                total_flux_dict[temp] += flux.sum(1)

        # Calculate flux fractions
        total_flux = np.asarray([v for k, v in total_flux_dict.items()]).sum(0)
        for temp in T_P + T_NP:
            flux_fraction_dict[temp] = total_flux_dict[temp] / total_flux

        # Write combined info
        info_dict_comb = dict()
        all_keys = ["A", "means", "vars", "skew", "tot_flux"]  # P: "A", NP: "means", "vars", "skew", "tot_flux"
        for key in all_keys:
            temp_dict = dict()
            for temp in T_P + T_NP:
                is_add_two = temp == ADD_template
                if is_add_two:
                    temp_keys = list(data_dict_0[temp]["info"].keys())
                else:
                    temp_keys = list(data_dict[temp]["info"].keys())

                if key in temp_keys:
                    if is_add_two:
                        temp_dict[temp] = np.vstack([data_dict_0[temp]["info"][key],
                                                     data_dict_1[temp]["info"][key]]).T
                    else:
                        temp_dict[temp] = data_dict[temp]["info"][key]

            info_dict_comb[key] = temp_dict

        # Store in "data_out" dictionary
        data_out["data"] = combined_map.T
        data_out["flux_fraction"] = flux_fraction_dict
        data_out["info"] = info_dict_comb

        # Now: compute histograms
        if do_any_hist:
            # Helper function to get the denominator to normalise the histograms: histogram sums whenever histogram is
            # not empty, otherwise: 1
            def get_denominator(hist_sum):
                denominator = np.ones_like(hist_sum)
                denominator[hist_sum > 0] = hist_sum[hist_sum > 0]
                return denominator


            data_out["hists"] = dict()

            for hist_template in hist_templates:
                is_add_two = hist_template == ADD_template
                dd_check = data_dict_0 if is_add_two else data_dict
                if is_add_two:
                    assert hist_template in data_dict_0.keys(), "Histogram template " + hist_template + " not found!"
                else:
                    assert hist_template in data_dict.keys(), "Histogram template " + hist_template + " not found!"
                n_maps_per_file = dd_check[hist_template]["data"].shape[0]
                data_out["hists"][hist_template] = dict()

                # dNdF histogram
                if do_dNdF:
                    if "flux_arr" not in dd_check[hist_template].keys():
                        raise RuntimeError("Error! GCE PS data does NOT contain lists with flux array! Aborting...")
                    if is_add_two:
                        hist_input = [np.hstack([data_dict_0[hist_template]["flux_arr"][i],
                                                 data_dict_1[hist_template]["flux_arr"][i]])
                                        for i in range(n_maps_per_file)]
                    else:
                        hist_input = data_dict[hist_template]["flux_arr"]
                    dNdF_hist = np.asarray([np.histogram(hist_input[i], weights=hist_input[i] ** power_of_F_dNdF,
                                                             bins=bins_dNdF)[0] for i in range(n_maps_per_file)])
                    dNdF_hist_sum = dNdF_hist.sum(1)
                    data_out["hists"][hist_template]["dNdF"] = dNdF_hist / np.expand_dims(get_denominator(dNdF_hist_sum), -1)

                # counts per PS histogram
                if do_counts_per_PS:
                    if "n_phot" not in dd_check[hist_template].keys():
                        raise RuntimeError("Error! GCE PS data does NOT contain lists with photon counts! Aborting...")
                    if is_add_two:
                        hist_input = [np.hstack([data_dict_0[hist_template]["n_phot"][i],
                                                 data_dict_1[hist_template]["n_phot"][i]])
                                        for i in range(n_maps_per_file)]
                    else:
                        hist_input = data_dict[hist_template]["n_phot"]
                    counts_per_PS_hist = np.asarray([np.histogram(hist_input[i], weights=hist_input[i],
                                                                  bins=bins_counts_per_PS)[0] for i in range(n_maps_per_file)])
                    counts_per_PS_hist_sum = counts_per_PS_hist.sum(1)
                    data_out["hists"][hist_template]["counts_per_PS"] = counts_per_PS_hist \
                                                                        / np.expand_dims(get_denominator(counts_per_PS_hist_sum), -1)

                # counts per pixel histogram
                if do_counts_per_pix:
                    if len(dd_check[hist_template]["data"].shape) != 3:
                        raise RuntimeError("Error! Data does NOT contain second channel with map before PSF application! Aborting...")
                    if is_add_two:
                        hist_input = np.stack([data_dict_0[hist_template]["data"], data_dict_1[hist_template]["data"]], 3).sum(3)
                    else:
                        hist_input = data_dict[hist_template]["data"]
                    counts_per_pix_hist = np.asarray([np.histogram(hist_input[i, :, 1], weights=hist_input[i, :, 1],
                                                                   bins=bins_counts_per_pix)[0] for i in range(n_maps_per_file)])
                    counts_per_pix_hist_sum = counts_per_pix_hist.sum(1)
                    data_out["hists"][hist_template]["counts_per_pix"] = counts_per_pix_hist / np.expand_dims(get_denominator(counts_per_pix_hist_sum), -1)

        # Save the data
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
        print(np.round(np.mean(combined_map.sum(1))))
        print(dash + "\n")

        # Stats concerning histograms
        for hist_template in hist_templates:
            print("Histogram stats for template", hist_template)
            if do_dNdF:
                print("  Mean dNdF histogram:", np.round(data_out["hists"][hist_template]["dNdF"].mean(0), 3))
            if do_counts_per_PS:
                print("  Mean counts per PS histogram:", np.round(data_out["hists"][hist_template]["counts_per_PS"].mean(0), 3))
                print("  Mean counts per PS histogram: fraction of maps with counts in highest bin: ",
                      np.mean(data_out["hists"][hist_template]["counts_per_PS"][:, -1] > 0))
            if do_counts_per_pix:
                print("  Mean counts per pix histogram:", np.round(data_out["hists"][hist_template]["counts_per_pix"].mean(0), 3))
                print("  Mean counts per pix histogram: fraction of maps with counts in highest bin: ",
                      np.mean(data_out["hists"][hist_template]["counts_per_pix"][:, -1] > 0))

        # Collect garbage to free memory
        auto_garbage_collect()

print(dash)
print("DONE!")
print(dash)

# Plot some histograms:
# i_plot = 0
# n_plot = 1
# plot_tmp = "gce_12_PS"
# [plt.bar(np.log10(((bins_dNdF[1:] + bins_dNdF[:-1]) / 2)[1:-1]), data_out["hists"][plot_tmp]["dNdF"][i][1:-1],
#         alpha=0.6, width=0.15) for i in range(i_plot, i_plot + n_plot)]
