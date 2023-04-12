"""
Combine template maps to a combined map and calculate the flux fractions.
Two maps of the SAME PS template can be randomly added for specified templates, simulating two populations.
Moreover, SCD histograms are generated.
This function needs to be run TWICE:
1. save_filenames = True: saves the filenames for training and test data
2. save_filenames = False: combines the template maps
"""
import numpy as np
import os
import sys
import pickle
import time
from .utils import auto_garbage_collect


def combine_template_maps(save_filenames, params, job_id=None, train_range=None, val_range=None, test_range=None,
                          verbose=False):
    """
    Combine template maps and save the combined maps.
    :param save_filenames: save filenames or combine maps (Step 1.: True, Step 2.: False)
    :param params: DotDict containing the settings (see parameters.py)
    :param job_id: if running several jobs for the data combination: ID of the current job
    :param train_range: [first index, last index] for training data handled by this job
    :param val_range: [first index, last index] for validation data handled by this job
    :param test_range: [first index, last index] for testing data handled by this job
    :param verbose: print some outputs
    """
    start_time = time.time()
    dash = 80 * "="

    # Get settings
    t_p = params.mod["models_P"]
    t_ps = params.mod["models_PS"]

    nside = params.data["nside"]
    input_path = params.gen["template_maps_folder"]
    output_path = params.gen["combined_maps_folder"]

    add_two_temps_ps = params.tt["add_two_temps_PS"]

    filename_out = params.comb["filename_base"]
    n_val = params.comb["N_val"]
    n_test = params.comb["N_test"]

    hist_templates = params.comb["hist_templates"]
    do_dndf = params.comb["do_dNdF"]
    do_counts_per_ps = params.comb["do_counts_per_PS"]
    do_counts_per_pix = params.comb["do_counts_per_pix"]
    bins_dndf = params.comb["bins_dNdF"]
    bins_counts_per_ps = params.comb["bins_counts_per_PS"]
    bins_counts_per_pix = params.comb["bins_counts_per_pix"]
    power_of_f_dndf = params.comb["power_of_F_dNdF"]

    i_channel_select = 1 if params.comb["combine_without_PSF"] else 0

    shuffle_random = True

    os.makedirs(output_path, exist_ok=True)
    do_any_hist = (do_dndf or do_counts_per_ps or do_counts_per_pix) and (len(hist_templates) > 0)
    files_dict_no_add_two = np.setdiff1d(t_p + t_ps, add_two_temps_ps)

    if save_filenames:
        if os.path.isfile(os.path.join(output_path, "filenames_combined_train.pickle")):
            print("File {:} already exists! Aborting...".format(
                os.path.join(output_path, "filenames_combined_train.pickle")))
            return
        if os.path.isfile(os.path.join(output_path, "filenames_combined_val.pickle")):
            print("File {:} already exists! Aborting...".format(
                os.path.join(output_path, "filenames_combined_val.pickle")))
            return
        if os.path.isfile(os.path.join(output_path, "filenames_combined_test.pickle")):
            print("File {:} already exists! Aborting...".format(
                os.path.join(output_path, "filenames_combined_test.pickle")))
            return

        train_files_dict = dict()
        val_files_dict = dict()
        test_files_dict = dict()
        settings_dict = dict()

        for temp in t_p + t_ps:
            # Get content of folder
            folder = os.path.join(input_path, temp)
            all_files = os.listdir(folder)
            all_files = np.asarray([file for file in all_files if "EXCLUDE" not in file])
            # Load settings dict
            settings_ind = np.argwhere(["settings" in file for file in all_files])[0][0]
            settings_file = open(os.path.join(folder, all_files[settings_ind]), 'rb')
            settings_dict[temp] = pickle.load(settings_file)
            settings_file.close()
            # Delete settings file from array
            all_files = np.delete(all_files, settings_ind)
            if temp in add_two_temps_ps:
                test_files = all_files[:2*n_test]
                val_files = all_files[2*n_test: 2*(n_test+n_val)]
                train_files = all_files[2*(n_test+n_val):]
            else:
                test_files = all_files[:n_test]
                val_files = all_files[n_test:(n_test+n_val)]
                train_files = all_files[n_test+n_val:]
            test_length = len(test_files)
            val_length = len(val_files)
            train_length = len(train_files)

            assert train_length >= 1 + (temp in add_two_temps_ps), \
                "There are no training files left for template " + temp + "..."

            # if requested: shuffle
            if shuffle_random:
                random_vec_train = np.random.choice(np.arange(train_length), train_length, replace=False)
                random_vec_val = np.random.choice(np.arange(val_length), val_length, replace=False)
                random_vec_test = np.random.choice(np.arange(test_length), test_length, replace=False)
                train_files = train_files[random_vec_train]
                val_files = val_files[random_vec_val]
                test_files = test_files[random_vec_test]

            train_files_dict[temp] = train_files
            val_files_dict[temp] = val_files
            test_files_dict[temp] = test_files

        # Save the file with the filename dictionary
        with open(os.path.join(output_path, "filenames_combined_train.pickle"), 'wb') as f1:
            pickle.dump(train_files_dict, f1)
            print("Filenames for training saved.")

        with open(os.path.join(output_path, "filenames_combined_val.pickle"), 'wb') as f2:
            pickle.dump(val_files_dict, f2)
            print("Filenames for validation saved.")

        with open(os.path.join(output_path, "filenames_combined_test.pickle"), 'wb') as f3:
            pickle.dump(test_files_dict, f3)
            print("Filenames for testing saved.")

        # Save the file with the settings dictionary
        with open(os.path.join(output_path, "settings_combined.pickle"), 'wb') as f4:
            pickle.dump(settings_dict, f4)
            print("Settings file saved.")
            print("EXITING.")
        return

    else:
        try:
            train_files_dict = pickle.load(open(os.path.join(output_path, "filenames_combined_train.pickle"), "rb"))
            val_files_dict = pickle.load(open(os.path.join(output_path, "filenames_combined_val.pickle"), "rb"))
            test_files_dict = pickle.load(open(os.path.join(output_path, "filenames_combined_test.pickle"), "rb"))
            settings_dict = pickle.load(open(os.path.join(output_path, "settings_combined.pickle"), "rb"))
        except (EOFError, IOError, FileNotFoundError):
            print("Run this script with save_filenames=True first!")
            sys.exit(1)

    # Get exposure and unmasked pixels
    assert train_files_dict.keys() == val_files_dict.keys() == test_files_dict.keys(), \
        "Training and testing dictionaries have different keys!"

    exp = np.unique(np.asarray([settings_dict[key]["exp"] for key in [*train_files_dict]]), axis=0).squeeze()
    rescale_compressed = np.unique(np.asarray([settings_dict[key]["rescale_compressed"]
                                               for key in [*train_files_dict]]), axis=0).squeeze()

    # Get number of files
    if len(files_dict_no_add_two) > 0:
        n_files_train = len(train_files_dict[files_dict_no_add_two[0]])
        n_files_val = len(val_files_dict[files_dict_no_add_two[0]])
        n_files_test = len(test_files_dict[files_dict_no_add_two[0]])
    else:
        n_files_train = len(train_files_dict[add_two_temps_ps[0]]) // 2
        n_files_val = len(val_files_dict[add_two_temps_ps[0]]) // 2
        n_files_test = len(test_files_dict[add_two_temps_ps[0]]) // 2

    # Set filenames of output files
    n_dig = int(np.ceil(np.log10(max(n_files_train, n_files_val, n_files_test))))
    files_train = np.asarray([filename_out + "_" + str(i).zfill(n_dig) + "_train.pickle" for i in range(n_files_train)])
    files_val = np.asarray([filename_out + "_" + str(i).zfill(n_dig) + "_val.pickle" for i in range(n_files_val)])
    files_test = np.asarray([filename_out + "_" + str(i).zfill(n_dig) + "_test.pickle" for i in range(n_files_test)])
    settings_filename = filename_out + "_settings.pickle"

    # Make folders
    os.makedirs(os.path.join(output_path, "Train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "Validation"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "Test"), exist_ok=True)

    # Set ranges of files to write
    start_train = 0 if train_range is None else train_range[0]
    start_val = 0 if val_range is None else val_range[0]
    start_test = 0 if test_range is None else test_range[0]
    end_train = n_files_train - 1 if train_range is None else train_range[1]
    end_val = n_files_val - 1 if val_range is None else val_range[1]
    end_test = n_files_test - 1 if test_range is None else test_range[1]

    # Do the following for the training and the test samples
    for tvt, files_dict in enumerate([train_files_dict, val_files_dict, test_files_dict]):

        # Set training / validation / testing specific settings
        if tvt == 0:
            n_files = n_files_train
            files = files_train
            outfolder = os.path.join(output_path, "Train")
            start, end = start_train, end_train
            print_str = "=== TRAINING DATA ==="

        elif tvt == 1:
            n_files = n_files_val
            files = files_val
            outfolder = os.path.join(output_path, "Validation")
            start, end = start_val, end_val
            print_str = "=== VALIDATION DATA ==="

        else:
            n_files = n_files_test
            files = files_test
            outfolder = os.path.join(output_path, "Test")
            start, end = start_test, end_test
            print_str = "=== TESTING DATA ==="

        # Check correctness
        if len(files_dict_no_add_two) > 0:
            assert np.all([len(files_dict[key]) == len(files_dict[files_dict_no_add_two[0]])
                           for key in files_dict_no_add_two]), \
                "The number of files for each model is not the same! Aborting..."

        # Write a combined settings dictionary (need to reorder: temp -> key => key -> temp)
        if tvt == 0:
            if int(job_id) == 0:
                settings_dict_comb = dict()
                all_keys = ["T", "T_corr", "priors", "max_NP_sources"]
                for key in all_keys:
                    temp_dict = dict()
                    for temp in t_p + t_ps:
                        temp_keys = list(settings_dict[temp].keys())
                        if key in temp_keys:
                            temp_dict[temp] = settings_dict[temp][key]
                    settings_dict_comb[key] = temp_dict
                settings_dict_comb["exp"] = exp
                settings_dict_comb["rescale_compressed"] = rescale_compressed
                settings_dict_comb["nside"] = nside
                indices_roi_all = np.asarray([settings_dict[temp]["indices_roi"] for temp in t_p + t_ps])
                indices_roi_unique = np.unique(indices_roi_all, axis=0)
                if indices_roi_unique.shape[0] > 1:
                    raise ValueError("ROI indices for the different template maps are not identical!")
                settings_dict_comb["indices_roi"] = indices_roi_unique.flatten()
                format_all = np.asarray([settings_dict[temp]["format"] for temp in t_p + t_ps])
                format_unique = np.unique(format_all)
                if len(format_unique) > 1:
                    raise ValueError("Different formats (RING / NEST) for the different templates are not supported!")
                settings_dict_comb["format"] = format_unique[0]

                # Histogram-specific settings
                if do_any_hist:
                    settings_dict_comb["hist_bins"] = dict()
                    if do_dndf:
                        settings_dict_comb["hist_bins"]["dNdF"] = bins_dndf
                        settings_dict_comb["hist_bins"]["power_of_F_dNdF"] = power_of_f_dndf
                    if do_counts_per_ps:
                        settings_dict_comb["hist_bins"]["counts_per_PS"] = bins_counts_per_ps
                    if do_counts_per_pix:
                        settings_dict_comb["hist_bins"]["counts_per_pix"] = bins_counts_per_pix

                if os.path.isfile(os.path.join(output_path, "Train", settings_filename)):
                    print("Settings file already exists...")
                else:
                    with open(os.path.join(output_path, "Train", settings_filename), 'wb') as f:
                        pickle.dump(settings_dict_comb, f)
                        print("Combined settings file written.")

        # in any case: need to get exp_indices_roi
        indices_roi_all = np.asarray([settings_dict[temp]["indices_roi"] for temp in t_p + t_ps])
        indices_roi_unique = np.unique(indices_roi_all, axis=0)
        if indices_roi_unique.shape[0] > 1:
            raise ValueError("ROI pixels for the different template maps are not identical!")
        exp_indices_roi = exp[indices_roi_unique.flatten()]

        # Print info
        if verbose:
            print(dash)
            print("NAME:",  params.comb["data_name"])
            print("job_id =", job_id)
            print(print_str)
            print("Starting to combine the maps {0} ... {1} out of {2} in total.".format(start, end, n_files))
            print("Poissonian models:", t_p)
            print("Point-source models:", t_ps)
            print("Each map will contain 2 template maps of templates", add_two_temps_ps, "!")

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
            combined_map = np.int32(0)

            for i_temp, temp in enumerate(t_p + t_ps):
                total_flux_dict[temp] = 0.0
                is_add_two = temp in add_two_temps_ps
                repeat_range = [0, 1] if is_add_two else [0]
                for rep in repeat_range:
                    if is_add_two:
                        this_file = files_dict[temp][2 * i_file + rep]
                    else:
                        this_file = files_dict[temp][i_file]
                    data_file = open(os.path.join(input_path, temp, this_file), 'rb')
                    temp_data = pickle.load(data_file)
                    temp_map = temp_data["data"]  # n_maps_per_chunk x n_indices_roi
                    if is_add_two:
                        if rep == 0:
                            data_dict_0[temp] = temp_data
                        else:
                            data_dict_1[temp] = temp_data
                        data_file.close()
                    else:
                        # Load data
                        data_dict[temp] = temp_data
                    # If PS map with second channel for data without PSF: extract desired channel
                    if len(temp_map.shape) == 3:
                        temp_map = temp_map[:, :, i_channel_select]
                    # Add to combined map
                    combined_map += temp_map
                    # Calculate flux
                    flux = temp_map / np.expand_dims(exp_indices_roi, 0)
                    # Total flux of template: sum over pixels
                    total_flux_dict[temp] += flux.sum(1)

            # Calculate flux fractions
            total_flux = np.asarray([v for k, v in total_flux_dict.items()]).sum(0)
            for temp in t_p + t_ps:
                flux_fraction_dict[temp] = total_flux_dict[temp] / total_flux

            # Write combined info
            info_dict_comb = dict()
            all_keys = ["A", "means", "vars", "skew", "tot_flux"]  # P: "A", NP: "means", "vars", "skew", "tot_flux"
            for key in all_keys:
                temp_dict = dict()
                for temp in t_p + t_ps:
                    is_add_two = temp in add_two_temps_ps
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
                # Helper function to get the denominator to normalise the histograms:
                # histogram sums whenever histogram is not empty, otherwise: 1
                def get_denominator(hist_sum):
                    denominator = np.ones_like(hist_sum)
                    denominator[hist_sum > 0] = hist_sum[hist_sum > 0]
                    return denominator

                data_out["hists"] = dict()

                for hist_template in hist_templates:
                    is_add_two = hist_template in add_two_temps_ps
                    dd_check = data_dict_0 if is_add_two else data_dict
                    if is_add_two:
                        assert hist_template in data_dict_0.keys(), \
                            "Histogram template " + hist_template + " not found!"
                    else:
                        assert hist_template in data_dict.keys(), \
                            "Histogram template " + hist_template + " not found!"
                    n_maps_per_file = dd_check[hist_template]["data"].shape[0]
                    data_out["hists"][hist_template] = dict()

                    # dNdF histogram
                    if do_dndf:
                        if "flux_arr" not in dd_check[hist_template].keys():
                            raise RuntimeError("Error! GCE PS data does NOT contain lists with flux array! Aborting...")
                        if is_add_two:
                            hist_input = [np.hstack([data_dict_0[hist_template]["flux_arr"][i],
                                                     data_dict_1[hist_template]["flux_arr"][i]])
                                            for i in range(n_maps_per_file)]
                        else:
                            hist_input = data_dict[hist_template]["flux_arr"]
                        dndf_hist = np.asarray([np.histogram(hist_input[i], weights=hist_input[i] ** power_of_f_dndf,
                                                                 bins=bins_dndf)[0] for i in range(n_maps_per_file)])
                        dndf_hist_sum = dndf_hist.sum(1)
                        data_out["hists"][hist_template]["dNdF"] = dndf_hist \
                                                                   / np.expand_dims(get_denominator(dndf_hist_sum), -1)

                    # counts per PS histogram
                    if do_counts_per_ps:
                        if "n_phot" not in dd_check[hist_template].keys():
                            raise RuntimeError("Error! GCE PS data does NOT contain photon count lists! Aborting!")
                        if is_add_two:
                            hist_input = [np.hstack([data_dict_0[hist_template]["n_phot"][i],
                                                     data_dict_1[hist_template]["n_phot"][i]])
                                            for i in range(n_maps_per_file)]
                        else:
                            hist_input = data_dict[hist_template]["n_phot"]
                        counts_per_ps_hist = np.asarray([np.histogram(hist_input[i], weights=hist_input[i],
                                                                      bins=bins_counts_per_ps)[0]
                                                         for i in range(n_maps_per_file)])
                        counts_per_ps_hist_sum = counts_per_ps_hist.sum(1)
                        data_out["hists"][hist_template]["counts_per_PS"] \
                            = counts_per_ps_hist / np.expand_dims(get_denominator(counts_per_ps_hist_sum), -1)

                    # counts per pixel histogram
                    if do_counts_per_pix:
                        if len(dd_check[hist_template]["data"].shape) != 3:
                            raise RuntimeError("Error! Data does NOT contain second channel"
                                               " with map before PSF application! Aborting...")
                        if is_add_two:
                            hist_input = np.stack([data_dict_0[hist_template]["data"],
                                                   data_dict_1[hist_template]["data"]], 3).sum(3)
                        else:
                            hist_input = data_dict[hist_template]["data"]
                        counts_per_pix_hist = np.asarray([np.histogram(hist_input[i, :, 1], weights=hist_input[i, :, 1],
                                                                       bins=bins_counts_per_pix)[0]
                                                          for i in range(n_maps_per_file)])
                        counts_per_pix_hist_sum = counts_per_pix_hist.sum(1)
                        data_out["hists"][hist_template]["counts_per_pix"] \
                            = counts_per_pix_hist / np.expand_dims(get_denominator(counts_per_pix_hist_sum), -1)

            # Save the data
            with open(os.path.join(outfolder, files[i_file]), 'wb') as f:
                pickle.dump(data_out, f)

            # Print some stats
            if verbose:
                print("File {0} / {1}:".format(i_file + 1, len(files)))
                print(dash)
                print("Number of simulations: {0}".format(data_out["data"].shape[1]))
                print("Templates:")
                print(list(data_out["flux_fraction"].keys()))
                print("Max. flux fraction for each template:")
                print([np.round(data_out["flux_fraction"][key].max(), 2)
                       for key in data_out["flux_fraction"].keys()])
                print("Min. flux fraction for each template:")
                print([np.round(data_out["flux_fraction"][key].min(), 2)
                       for key in data_out["flux_fraction"].keys()])
                print("Mean flux fraction for each template:")
                print([np.round(data_out["flux_fraction"][key].mean(), 2)
                       for key in data_out["flux_fraction"].keys()])
                print("Median flux fraction for each template:")
                print([np.round(np.median(data_out["flux_fraction"][key]), 2)
                       for key in data_out["flux_fraction"].keys()])
                print("Avg. total number of counts:")
                print(np.round(np.mean(combined_map.sum(1))))
                print(dash + "\n")

                # Stats concerning histograms
                for hist_template in hist_templates:
                    print("Histogram stats for template", hist_template)
                    if do_dndf:
                        print("  Mean dNdF histogram:", np.round(data_out["hists"][hist_template]["dNdF"].mean(0), 3))
                    if do_counts_per_ps:
                        print("  Mean counts per PS histogram:",
                              np.round(data_out["hists"][hist_template]["counts_per_PS"].mean(0), 3))
                        print("  Mean counts per PS histogram: fraction of maps with counts in highest bin: ",
                              np.mean(data_out["hists"][hist_template]["counts_per_PS"][:, -1] > 0))
                    if do_counts_per_pix:
                        print("  Mean counts per pix histogram:",
                              np.round(data_out["hists"][hist_template]["counts_per_pix"].mean(0), 3))
                        print("  Mean counts per pix histogram: fraction of maps with counts in highest bin: ",
                              np.mean(data_out["hists"][hist_template]["counts_per_pix"][:, -1] > 0))

            # Collect garbage to free memory
            auto_garbage_collect()

    print(dash)
    print("Done! Computation took {:.2} seconds.".format(time.time() - start_time))
    print(dash)
