"""
File that shows how to open the compressed data and obtain the maps.
"""
import numpy as np
import healpy as hp
import pickle
import os

# Only unmasked indices -> all indices
def masked_to_full(x, unmasked_pix, fill_value=0.0, npix=None, nside=None):
    """
    Return a full map (that is consisting of npix pixels) with values of x in pixels given by unmasked_pix.
    NOTE: Make sure that "unmasked_pix" use the same format (RING/NEST) as the values x!
    :param x: values
    :param unmasked_pix: pixels that shall be filled with x
    :param fill_value: fill value for OTHER pixels (that are not in unmasked_pix)
    :param npix: either specify healpix npix
    :param nside: OR specify healpix nside
    :return: all-sky map
    """
    if npix is None and nside is not None:
        npix = hp.nside2npix(nside)
    elif npix is None and nside is None:
        print("Error! No npix or nside provided.")
        os._exit(1)
    elif npix is not None and nside is not None:
        print("Warning! npix and nside provided! Using npix...")
    if len(x.shape) > 2:
        raise NotImplementedError

    out = np.ones((x.shape[0], npix)) * fill_value if len(x.shape) > 1 else np.ones(npix) * fill_value
    if len(x.shape) > 1:
        out[:, unmasked_pix] = x
    else:
        out[unmasked_pix] = x
    return out

data_file = "/home/flo/Documents/Latex/GCE/Data_for_paper/GCE_and_background_12_const_exp/GCE_and_background_12_const_exp_subset_256.pickle"
mask_file = "/home/flo/PycharmProjects/GCE/unmasked_pix.npy"

# Load settings file
unmasked_pix = np.load(mask_file)

# Load raw data file
data_pickle = open(data_file, 'rb')
data_dict = pickle.load(data_pickle)
data_pickle.close()

# Get maps
maps = masked_to_full(data_dict["data"].T, unmasked_pix=unmasked_pix, nside=128)

