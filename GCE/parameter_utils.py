import warnings
import os
import numpy as np
import pickle
from .utils import DotDict


def get_subdict(params, keys, return_normal_dict=True, delete_functions=False):
    """
    Get a subdictionary only containing desired parameters.
    :param params: parameter dictionary
    :param keys: keys to extract
    :param return_normal_dict: if True: return python dict instead of DotDict
    :param delete_functions: if True: deletes callable objects to make the output dict serializable
    (requires return_normal_dict == True)
    :return: subdictionary
    """
    subdict = DotDict()
    for k in keys:
        if k not in params.keys():
            raise KeyError("Key '{:}' not found in parameter dictionary!".format(k))
        subdict[k] = params[k]
    if return_normal_dict:
        return subdict.convert_to_dict(delete_functions=delete_functions)
    else:
        return subdict


def load_params_from_pickle(folder, return_normal_dict=False):
    """
    Load parameters from pickle file.
    :param folder: folder name
    :param return_normal_dict: return a python dict instead of DotDict
    :return: parameter dictionary
    """
    folder_content = os.listdir(folder)
    pickle_files = [f for f in folder_content if "params" in f and f.endswith(".pickle")]
    if len(pickle_files) == 0:
        raise FileNotFoundError("No parameter file found in folder '{:}'!".format(folder))
    elif len(pickle_files) == 1:
        param_file = pickle_files[0]
    else:
        warnings.warn("Warning! More than one parameter file found in folder '{:}'!"
                      " Loading the most recent one...".format(folder))
        md_times = [os.path.getmtime(os.path.join(folder, f)) for f in pickle_files]
        param_file = pickle_files[np.argmax(md_times)]

    with open(os.path.join(folder, param_file), "rb") as pf:
        param_dict = pickle.load(pf)

    if return_normal_dict:
        return param_dict
    else:
        return DotDict(param_dict)
