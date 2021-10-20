import numpy as np
import os
import sys
import pickle
from ..data_utils import dict_to_array


# # # # # # # # # # PairGenerator classes # # # # # # # # # #
class PairGenerator(object):
    """Base PairGenerator class"""

    def __init__(self, params, tvt):
        """
        Initialise the generator.
        :param params: parameter dictionary
        :param tvt: 0: training, 1: validation, 2: testing
        """
        self._p = params
        self.train_val_test = tvt


# CNN: PairGenerator class for pre-generated data
class PairGeneratorCNNPreGenerated(PairGenerator):
    def __init__(self, params, tvt, settings_dict=None):
        """
        Initialise the generator.
        :param params: parameter dictionary
        :param tvt: 0: training, 1: validation, 2: testing
        :param settings_dict: for validation and testing data: settings dict can be passed here
        """
        super(PairGeneratorCNNPreGenerated, self).__init__(params, tvt)

        if tvt == 0:
            subfolder_str = "Train"
        elif tvt == 1:
            subfolder_str = "Validation"
        elif tvt == 2:
            subfolder_str = "Test"
        else:
            raise NotImplementedError

        folder = os.path.join(self._p.gen["combined_maps_folder"], subfolder_str)
        assert os.path.exists(folder), "Folder '{:}' does not exist! Aborting...".format(folder)

        all_files = os.listdir(folder)
        all_files = [file for file in all_files if "EXCLUDE" not in file]  # Don't include files containing "EXCLUDE"

        # Setting file should be stored with training data
        if tvt == 0:
            try:
                settings_ind = np.argwhere(["settings" in file for file in all_files])[0][0]
                settings_file = open(os.path.join(folder, all_files[settings_ind]), 'rb')
                self.settings_dict = pickle.load(settings_file)
                settings_file.close()

            except (FileNotFoundError, EOFError, IOError):
                print("Opening settings file failed. Aborting...")
                sys.exit(1)

        else:
            self.settings_dict = settings_dict
            settings_ind = -1

        self._all_files = [file for (file, i) in zip(all_files, range(len(all_files))) if i != settings_ind]
        self._data_folder = folder
        self._index_in_array = 0
        self._file_no = 0
        self._active_file = ""
        self._data_dict = {}
        self._cond_must_be_imposed = False

    def get_next_pair(self, verbose=1, extra_info=False):
        """
        :param verbose: verbosity level (0 - 3)
        :param extra_info: also return extra_info field?
        :return: next data - label pair (if EOF is reach: load from new file, if all files have been used: repeat)
        """
        while True:
            ready_to_yield = False
            if self.train_val_test == 0:
                print_str = "Training:"
            elif self.train_val_test == 1:
                print_str = "Validation:"
            elif self.train_val_test == 2:
                print_str = "Testing:"
            else:
                raise NotImplementedError

            while not ready_to_yield:
                # Get first file
                if len(self._active_file) == 0:
                    self._active_file = self._all_files[self._file_no]
                    self._index_in_array = 0
                    if verbose == 2:
                        print(print_str, "Opening file", self._active_file, "...")
                    try:
                        data_file = open(os.path.join(self._data_folder, self._active_file), 'rb')
                        self._data_dict = pickle.load(data_file)
                        data_file.close()
                    except (FileNotFoundError, EOFError, IOError):
                        print("Opening file", self._active_file, "failed. Aborting...")
                        sys.exit(1)

                # If end of a file
                elif self._index_in_array == self._data_dict["data"].shape[1]:
                    self._index_in_array = 0
                    self._file_no += 1
                    # If through all files: start at file 0
                    if self._file_no >= len(self._all_files):
                        self._file_no = 0
                        if verbose >= 1:
                            print(print_str, "End of epoch!")
                    self._active_file = self._all_files[self._file_no]
                    if verbose == 2:
                        print(print_str, "Opening file", self._active_file, "...")
                    try:
                        data_file = open(os.path.join(self._data_folder, self._active_file), 'rb')
                        self._data_dict = pickle.load(data_file)
                        data_file.close()
                    except (FileNotFoundError, EOFError, IOError):
                        print("Opening file", self._active_file, "failed. Aborting...")
                        sys.exit(1)

                # Initialise labels
                n_labels = 2 if self._p.nn.hist["return_hist"] else 1
                labels = [None] * n_labels

                # Calculate FFs (if exposure correction is removed, this is the fraction of flux rather than the counts)
                labels[0] = np.asarray([self._data_dict["flux_fraction"][key][self._index_in_array]
                                        for key in self._p.mod["models"]])

                template_map_counts = self._data_dict["data"][:, self._index_in_array]
                new_array = template_map_counts

                # Get histogram
                if self._p.nn.hist["return_hist"]:
                    try:
                        hict_scd = np.asarray([self._data_dict["hists"][temp]
                                               [self._p.nn.hist["which_histogram"]][self._index_in_array, :]
                                               for temp in self._p.nn.hist["hist_templates"]]).T
                    except KeyError as e:
                        print("No histogram data found or histogram data corrupted! Aborting...")
                        raise e

                    labels[1] = hict_scd

                # Check if conditions are satisfied (training data only)
                ready_to_yield = True
                if self.train_val_test == 0 and (self._p.nn.cond["cond_on_training_data"] is not None
                                                 or self._p.nn.cond["cond_on_training_labels"] is not None):
                    impose_cond = self._cond_must_be_imposed \
                                  or (self._p.nn.cond["prob_for_conditions"] > np.random.uniform(0, 1, 1)[0])
                    if impose_cond:
                        label_cond_result = True if self._p.nn.cond["cond_on_training_labels"] is None \
                            else self._p.nn.cond["cond_on_training_labels"](labels)
                        map_cond_result = True if self._p.nn.cond["cond_on_training_data"] is None \
                            else self._p.nn.cond["cond_on_training_data"](new_array)
                        ready_to_yield = label_cond_result and map_cond_result

                # Break
                if ready_to_yield:
                    break
                else:
                    self._cond_must_be_imposed = True
                    self._index_in_array += 1

            # Build the output dictionary
            if verbose >= 3:
                print("Yielding file", self._active_file, "index", self._index_in_array)
            output_dict = {"data": new_array.astype(np.float32), "label": tuple(labels)}

            if np.any(extra_info):
                # NOTE: The order of the models in the extra info dict. is NOT the same as in self._p.mod["models"]!
                # This is because not all the fields are present for all the templates!
                # When using the extra info, make sure you know what you're looking at!
                extra_info_dict = {
                    info_key: np.asarray([self._data_dict["info"][info_key][model_key][self._index_in_array]
                                          for model_key in self._data_dict["info"][info_key].keys()])
                    for info_key in self._data_dict["info"].keys()}
                extra_info = dict_to_array(extra_info_dict)
                output_dict["extra_info"] = extra_info.astype(np.float32)

            # Yield the sample
            yield output_dict

            self._index_in_array += 1
            self._cond_must_be_imposed = False
