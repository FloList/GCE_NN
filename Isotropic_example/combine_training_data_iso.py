"""
This combines data per model to a combined map and calculates the flux fractions.
Moreover, histograms are generated from the n_phot lists.
This script needs to be run TWICE:
1. SAVE_FILENAMES = True: saves the filenames for training and test data
2. SAVE_FILENAMES = False: combines the template maps
NOTE: in contrast to the other combining functions, NO MASKS can be applied here and the ROI can NOT be changed!
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
SAVE_FILENAMES = True  # first step: True: save filenames to combine, second step: False: generate maps
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
nside = 256 if HPC else 128
input_path = '/scratch/u95/fl9575/GCE_v2/data/Iso_example_maps' if HPC \
         else '/home/flo/PycharmProjects/GCE/data/Iso_example_maps'
input_path += "_" + str(nside)
name = "Iso_maps_combined"  # folder name for combined data
filename_out = "Iso_maps"  # filename basis for combined maps
output_path = '/scratch/u95/fl9575/GCE_v2/data/' + name if HPC else '/home/flo/PycharmProjects/GCE/data/' + name
training_name = "Train"
testing_name = "Test"
shuffle_random = True
npix = hp.nside2npix(nside)
allow_different_names = True  # if False: give an error message if the file names for the models are not identical
extend_to_length = None  # if not None: reuse files in order to generate "extend_to_length" random combinations of
                       # the maps from the individual models (only for training data)
N_test = 2500  # number of files for the test set (have NO contribution in common with training set)


# Histogram settings
do_dNdF = True  # save histograms of dNdFs?
do_counts_per_PS = False  # save histograms of counts per PS?
do_counts_per_pix = False  # save histograms of counts per pixel before the PSF (as saved in the 2nd data channel)?
# Define the histogram bins
bins_dNdF = np.asarray([-np.infty] + list(np.logspace(-1.5, 2, 21)) + [np.infty])
power_of_F_dNdF = 1  # power of F to multiply dN/dF with
bins_counts_per_PS = bins_counts_per_pix = None
# Hist template name
hist_templates = ["iso_PS"]

#############
# CHOOSE MODELS TO COMBINE
T_P = []
T_NP = ["iso_PS"]
#############
# Create the output folder (if it doesn't exist yet)
mkdir_p(output_path)
dash = 80 * "="
do_any_hist = (do_dNdF or do_counts_per_PS or do_counts_per_pix) and (len(hist_templates) > 0)

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
        print("Run this script with SAVE_FILENAMES=True first!")
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
            data_file = open(os.path.join(input_path, temp, files_dict[temp][i_file]), 'rb')
            # Load data
            data_dict[temp] = pickle.load(data_file)
            data_file.close()
            # Get template map
            temp_map = data_dict[temp]["data"]  # n_maps_per_chunk x n_unmasked_pixs
            # If PS map with second channel for data without PSF
            if len(temp_map.shape) == 3:
                temp_map = temp_map[:, :, 0]
            # Add to combined map
            combined_map += temp_map
            # Calculate flux
            flux = temp_map / np.expand_dims(exp_unmasked_pix, 0)
            # Total flux of template: sum over pixels
            total_flux_dict[temp] = flux.sum(1)

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
                temp_keys = list(data_dict[temp]["info"].keys())
                if key in temp_keys:
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
                assert hist_template in data_dict.keys(), "Histogram template " + hist_template + " not found!"
                n_maps_per_file = data_dict[hist_template]["data"].shape[0]
                data_out["hists"][hist_template] = dict()

                # dNdF histogram
                if do_dNdF:
                    if not "flux_arr" in data_dict[hist_template].keys():
                        raise RuntimeError("Error! GCE PS data does NOT contain lists with flux array! Aborting...")
                    dNdF_hist = np.asarray([np.histogram(data_dict[hist_template]["flux_arr"][i],
                                                             weights=data_dict[hist_template]["flux_arr"][i] ** power_of_F_dNdF,
                                                             bins=bins_dNdF)[0] for i in range(n_maps_per_file)])
                    dNdF_hist_sum = dNdF_hist.sum(1)

                    data_out["hists"][hist_template]["dNdF"] = dNdF_hist / np.expand_dims(get_denominator(dNdF_hist_sum), -1)

                # counts per PS histogram
                if do_counts_per_PS:
                    if not "n_phot" in data_dict[hist_template].keys():
                        raise RuntimeError("Error! GCE PS data does NOT contain lists with photon counts! Aborting...")
                    counts_per_PS_hist = np.asarray([np.histogram(data_dict[hist_template]["n_phot"][i],
                                                         weights=data_dict[hist_template]["n_phot"][i],
                                                         bins=bins_counts_per_PS)[0] for i in range(n_maps_per_file)])
                    counts_per_PS_hist_sum = counts_per_PS_hist.sum(1)
                    assert np.all(counts_per_PS_hist_sum == data_dict[hist_template]["data"][:, :, 0].sum(1)), \
                        "Sum of counts in histograms do not match counts in maps! Aborting..."
                    data_out["hists"][hist_template]["counts_per_PS"] = counts_per_PS_hist \
                                                                        / np.expand_dims(get_denominator(counts_per_PS_hist_sum), -1)

                # counts per pixel histogram
                if do_counts_per_pix:
                    if len(data_dict[hist_template]["data"].shape) != 3:
                        raise RuntimeError("Error! Data does NOT contain second channel with map before PSF application! Aborting...")
                    counts_per_pix_hist = np.asarray([np.histogram(data_dict[hist_template]["data"][i, :, 1],
                                                         weights=data_dict[hist_template]["data"][i, :, 1],
                                                         bins=bins_counts_per_pix)[0] for i in range(n_maps_per_file)])
                    counts_per_pix_hist_sum = counts_per_pix_hist.sum(1)
                    data_out["hists"][hist_template]["counts_per_pix"] = counts_per_pix_hist \
                                                                         / np.expand_dims(get_denominator(counts_per_pix_hist_sum), -1)

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
i_plot = 0
n_plot = 5
[plt.bar(np.log10(((bins_dNdF[1:] + bins_dNdF[:-1]) / 2)[1:-1]),
         data_out["hists"][hist_template]["dNdF"][i][1:-1], alpha=0.8, width=0.15)
 for i in range(i_plot, i_plot + n_plot)]
