###############################################################################
# masking functions taken from
# https://github.com/bsafdi/NPTFit/blob/master/NPTFit/create_mask.py
###############################################################################
import tensorflow as tf
import numpy as np
import healpy as hp
import os
from .utils import DotDict
from .pdf_sampler import PDFSampler


def make_plane_mask(band_mask_range, nside):
    """
    Masks within |b| < band_mask_range.
    :param band_mask_range: band mask range
    :param nside: nside resolution parameter
    :return: boolean mask
    """
    mask_none = np.arange(hp.nside2npix(nside))
    return (np.radians(90-band_mask_range) < hp.pix2ang(nside, mask_none)[0]) * \
           (hp.pix2ang(nside, mask_none)[0] < np.radians(90+band_mask_range))


def make_long_mask(l_deg_min, l_deg_max, nside):
    """
    Masks outside l_deg_min < l < l_deg_max
    :param l_deg_min: min. longitude
    :param l_deg_max: max. longitude
    :param nside: nside resolution parameter
    :return: boolean mask
    """
    mask_none = np.arange(hp.nside2npix(nside))
    return (np.radians(l_deg_max) < hp.pix2ang(nside, mask_none)[1]) * \
           (hp.pix2ang(nside, mask_none)[1] < np.radians(360 + l_deg_min))


def make_lat_mask(b_deg_min, b_deg_max, nside):
    """
    Masks outside b_deg_min < b < b_deg_max
    :param b_deg_min: min. latitude
    :param b_deg_max: max. latitude
    :param nside: nside resolution parameter
    :return: boolean mask
    """
    mask_none = np.arange(hp.nside2npix(nside))
    return np.logical_not(
           (np.radians(90-b_deg_max) < hp.pix2ang(nside, mask_none)[0]) *
           (hp.pix2ang(nside, mask_none)[0] < np.radians(90-b_deg_min)))


def make_ring_mask(inner, outer, ring_b, ring_l, nside):
    """
    Masks outside inner < r < outer, of a ring centred at (ring_b,ring_l)
    :param inner: inner radius
    :param outer: outer radius
    :param ring_b: b of ring centre
    :param ring_l: l of ring centre
    :param nside: nside resolution parameter
    :return: boolean mask
    """
    mask_none = np.arange(hp.nside2npix(nside))
    return np.logical_not(
           (np.cos(np.radians(inner)) >=
            np.dot(hp.ang2vec(np.radians(90-ring_b),
                   np.radians(ring_l)), hp.pix2vec(nside, mask_none))) *
           (np.dot(hp.ang2vec(np.radians(90-ring_b),
            np.radians(ring_l)), hp.pix2vec(nside, mask_none)) >=
            np.cos(np.radians(outer))))


def make_mask_total(nside=128,
                    band_mask=False, band_mask_range=30,
                    l_mask=False, l_deg_min=-30, l_deg_max=30,
                    b_mask=False, b_deg_min=-30, b_deg_max=30,
                    mask_ring=False, inner=0, outer=30,
                    ring_b=0, ring_l=0,
                    custom_mask=None):
    """
    Combines band, l, b, ring, and custom masks into a single mask
    :param nside: nside resolution parameter
    :param band_mask: apply band mask?
    :param band_mask_range: band mask range for plane mask
    :param l_mask: apply longitude mask?
    :param l_deg_min: min. longitude
    :param l_deg_max: max. longitude
    :param b_mask: apply latitude mask?
    :param b_deg_min: min. latitude
    :param b_deg_max: max. latitude
    :param mask_ring: apply ring mask?
    :param inner: inner radius
    :param outer: outer radius
    :param ring_b: b of ring centre
    :param ring_l: l of ring centre
    :param custom_mask: custom mask can be applied in addition
    :return: boolean mask
    """

    # Initialise an array where no pixels are masked
    mask_array = np.zeros(nside**2*12, dtype=bool)

    # Add masks depending on input
    if band_mask:
        mask_array += make_plane_mask(band_mask_range, nside)

    if l_mask:
        mask_array += make_long_mask(l_deg_min, l_deg_max, nside)

    if b_mask:
        mask_array += make_lat_mask(b_deg_min, b_deg_max, nside)

    if mask_ring:
        mask_array += make_ring_mask(inner, outer, ring_b, ring_l, nside)

    if custom_mask is not None:
        mask_array += custom_mask

    return mask_array


def get_template(fermi_folder, temp):
    """
    Returns a template.
    :param fermi_folder: folder containing the template maps
    :param temp: short name of template
    :return: template
    """
    if temp == "iso":
        t = np.load(os.path.join(fermi_folder, 'template_iso.npy'))
    elif temp == "dif":
        t = np.load(os.path.join(fermi_folder, 'template_dif.npy'))
    elif temp == "bub":
        t = np.load(os.path.join(fermi_folder, 'template_bub.npy'))
    elif temp == "bub_var":
        try:
            t = np.load(os.path.join(fermi_folder, 'template_nbub.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_bub_alt.npy'))
    elif temp == "gce":
        try:
            t = np.load(os.path.join(fermi_folder, 'template_gce.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_nfw_g1p0.npy'))
    elif temp == "gce_12":
        try:
            t = np.load(os.path.join(fermi_folder, 'template_gce_gamma_1.20.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_nfw_g1p2.npy'))
    elif temp == "gce_12_N":
        t = np.load(os.path.join(fermi_folder, 'template_gce_gamma_1.20_N.npy'))
    elif temp == "gce_12_S":
        t = np.load(os.path.join(fermi_folder, 'template_gce_gamma_1.20_S.npy'))
    elif temp == "disk":
        t = np.load(os.path.join(fermi_folder, 'template_dsk.npy'))
    elif temp == "thin_disk":
        try:
            t = np.load(os.path.join(fermi_folder, 'template_disk_r_s_5_z_s_0.3.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_dsk_z0p3.npy'))
    elif temp == "thick_disk":
        try:
            t = np.load(os.path.join(fermi_folder, 'template_disk_r_s_5_z_s_1.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_dsk_z1p0.npy'))
    elif temp == "dif_O_pibs":
        try:
            t = np.load(os.path.join(fermi_folder, 'ModelO_r25_q1_pibrem.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_Opi.npy'))
    elif temp == "dif_O_ic":
        try:
            t = np.load(os.path.join(fermi_folder, 'ModelO_r25_q1_ics.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_Oic.npy'))
    elif temp == "dif_A_pibs":
        t = np.load(os.path.join(fermi_folder, 'template_Api.npy'))
    elif temp == "dif_A_ic":
        t = np.load(os.path.join(fermi_folder, 'template_Aic.npy'))
    elif temp == "dif_F_pibs":
        t = np.load(os.path.join(fermi_folder, 'template_Fpi.npy'))
    elif temp == "dif_F_ic":
        t = np.load(os.path.join(fermi_folder, 'template_Fic.npy'))
    elif temp == "psc_3":
        try:
            t = np.load(os.path.join(fermi_folder, 'template_psc.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'template_psc_3fgl.npy'))
    elif temp == "psc_4":
        t = np.load(os.path.join(fermi_folder, 'template_psc_4fgl.npy'))
    elif temp == "3FGL_mask":
        try:
            t = np.load(os.path.join(fermi_folder, 'fermidata_pscmask.npy'))
        except FileNotFoundError:
            t = np.load(os.path.join(fermi_folder, 'fermidata_pscmask_3fgl.npy'))
    elif temp == "4FGL_mask":
        t = np.load(os.path.join(fermi_folder, 'fermidata_pscmask_4fgl.npy'))
    elif temp == "exp":
        t = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
    elif temp == "counts":
        t = np.load(os.path.join(fermi_folder, "fermidata_counts.npy"))
    elif temp == "fermi_map":
        t = np.load(os.path.join(fermi_folder, "fermidata_counts.npy"))
    else:
        raise NotImplementedError("Template", temp, "not available!")
    return t


def fermi_psf(r):
    """
    Fermi point-spread function.
    :param r: distance in radians from the centre of the point source
    :return: Fermi PSF
    """
    # Define parameters that specify the Fermi-LAT PSF at 2 GeV
    fcore = 0.748988248179
    score = 0.428653790656
    gcore = 7.82363229341
    stail = 0.715962650769
    gtail = 3.61883748683
    spe = 0.00456544262478

    # Define the full PSF in terms of two King functions
    def king_fn(x, sigma, gamma):
        return 1. / (2. * np.pi * sigma ** 2.) * (1. - 1. / gamma) * (1. + (x ** 2. / (2. * gamma * sigma ** 2.))) ** (
            -gamma)

    def fermi_psf_inner(r_):
        return fcore * king_fn(r_ / spe, score, gcore) + (1 - fcore) * king_fn(r_ / spe, stail, gtail)

    return fermi_psf_inner(r)


def get_fermi_pdf_sampler(n_points_f=int(1e6)):
    f = np.linspace(0, np.pi, n_points_f)
    pdf_psf = f * fermi_psf(f)
    pdf = PDFSampler(f, pdf_psf)
    return pdf


def masked_to_full(x, unmasked_pix, fill_value=0.0, npix=None, nside=None):
    """
    Return a full map (that is, consisting of npix pixels) with values of x in pixels given by unmasked_pix.
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
        raise RuntimeError("Error! No npix or nside provided.")
    elif npix is not None and nside is not None:
        print("Warning! npix and nside provided! Using npix and ignoring nside...")
    if len(x.shape) > 2:
        raise NotImplementedError

    out = np.ones((x.shape[0], npix)) * fill_value if len(x.shape) > 1 else np.ones(npix) * fill_value
    if len(x.shape) > 1:
        out[:, unmasked_pix] = x
    else:
        out[unmasked_pix] = x
    return out


def get_pixels(mask, nsides):
    """
    Find pixels in ROI (described by mask) for all values of nsides, such that the entire ROI is contained within the
    output pixels for all the values of nsides.
    For example, if all the unmasked values are contained within the inner pixel at nside = 1, this function will return
    all the pixels at each nside that lie within the coarse nside=1 pixel around the GC.
    :param mask: array that is 1 for all pixels that shall be masked, 0 else
    :param nsides: list of nsides
    :return: list of pixels for each entry of nsides
    """
    pixels = [None] * len(nsides)
    pixels[-1] = np.argwhere(1 - mask).flatten()

    for i in range(len(nsides) - 2, -1, -1):
        mask = np.asarray(hp.ud_grade(1.0 * mask, nside_out=nsides[i], order_in='NEST', order_out='NEST'))
        pixels[i] = np.argwhere(1 - mask).flatten()

    return pixels


def get_pixels_with_holes(mask, nsides):
    """
    Function that returns the pixels and allows for holes.
    Then, the NN needs to take care of these holes!
    In contrast to get_pixels, this function builds the indices top -> down.
    :params mask: boolean mask of the ROI
    :params nsides: list of nsides
    :return list of pixels for each entry of nsides
    """
    pixels = [None] * len(nsides)
    pixels[0] = np.argwhere(1 - mask).flatten()
    for i in range(1, len(nsides)):
        mask = hp.ud_grade(1.0 * mask, nside_out=nsides[i], order_in='NEST', order_out='NEST')
        mask[mask < 1] = 0
        pixels[i] = np.argwhere(1 - mask).flatten()
    return pixels


def build_index_dict(params):
    """
    Build a dictionary containing
    1. a list of the indices of the ROI pixels at each hierarchy level, from the selected nside downwards to 1
    2. a list of the indices of the ROI pixels at each hierarchy level such that the nside=1 pixel is contained
    at each hierarchy level.
    3. a list mapping the ROI indices to the extended ROI
    NOTE: NESTED FORMAT!
    :param params: parameter dictionary
    :return: dictionary containing information about the ROI
    """

    # Set up the mask for the ROI
    inner_band = params.data["inner_band"]
    outer_rad = params.data["outer_rad"]
    nside = params.data["nside"]
    nsides = params.nn.arch["nsides"]
    roi = make_mask_total(band_mask=True, band_mask_range=inner_band, mask_ring=True, inner=0,
                          outer=outer_rad, nside=nside)
    if params.data["mask_type"] == "3FGL":
        roi = (1 - (1 - roi) * (1 - get_template(params.gen["fermi_folder"], "3FGL_mask"))).astype(bool)
    elif params.data["mask_type"] == "4FGL":
        roi = (1 - (1 - roi) * (1 - get_template(params.gen["fermi_folder"], "4FGL_mask"))).astype(bool)
    roi = hp.reorder(roi, r2n=True)

    roi_extended = hp.reorder(make_mask_total(nside=1, mask_ring=True, inner=0,
                                              outer=outer_rad), r2n=True)
    roi_dict = dict()
    roi_dict["indexes"] = get_pixels_with_holes(roi, nsides)
    roi_dict["indexes_extended"] = get_pixels(roi_extended, nsides)
    roi_dict["ind_holes_to_ex"] = [np.asarray([np.argwhere(roi_dict["indexes_extended"][i] == ind)[0][0]
                                                for ind in roi_dict["indexes"][i]])
                                    for i in range(len(roi_dict["indexes"]))]

    return roi_dict


def dict_to_array(dic):
    """
    Convert a dictionary to an array to make it suitable for the tensorflow pipeline.
    :param dic: dictionary
    :return: list
    """
    def flatten(*n): return (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    return np.asarray(list(flatten([np.asarray(dic[key]).tolist() for key in [*dic]])))


def set_batch_dim_recursively(d, bs, lists_to_tuples=False):
    """
    Traverses the dictionary recursively and extracts the first bs elements along the batch dimension
    :param d: dictionary containing "data" and "labels"
    :param bs: batch size
    :param lists_to_tuples: convert all occurring lists to tuples along the way
    :return: dictionary with desired batch size
    """
    assert isinstance(d, (dict, DotDict)), "'d' must be a dictionary!"
    out_dict = {} if isinstance(d, dict) else DotDict()
    for k in d.keys():
        if isinstance(d[k], np.ndarray) or tf.is_tensor(d[k]):
            out_dict[k] = d[k][:bs]
        elif isinstance(d[k], (list, tuple)):
            out_dict[k] = [None] * len(d[k])
            for elem in range(len(d[k])):
                out_dict[k][elem] = d[k][elem][:bs]
            if lists_to_tuples:
                out_dict[k] = tuple(out_dict[k])
        else:
            raise NotImplementedError
    return out_dict


def split_into_n_and_s(template, nside=128, filename="template"):
    """
    Split up a template into two templates consisting of the N / S hemispheres only. NOTE: RING format!
    :param template: map name
    :param nside: nside parameter
    :param filename: output filename (if None: only return north and south maps and do not save them)
    :return: template map north, template map south
    """
    npix = hp.nside2npix(nside)
    pix_n = np.argwhere(hp.pix2vec(nside, range(npix))[-1] >= 0).flatten()
    pix_s = np.argwhere(hp.pix2vec(nside, range(npix))[-1] < 0).flatten()
    template_n, template_s = np.copy(template), np.copy(template)
    template_n[pix_s] = 0
    template_s[pix_n] = 0
    assert np.all(template_s + template_n == template), "Something went wrong! Aborting..."
    if filename is not None:
        np.save(filename + "_N", template_n)
        np.save(filename + "_S", template_s)
    return template_n, template_s


def get_fermi_counts(params, indexes_top, rescale_compressed=None, only_hemisphere=None):
    """
    Returns the counts in the Fermi map after the same pre-processing as for the training data.
    :param params: parameter dictionary
    :param indexes_top: indexes of ROI at highest considered resolution
    :param rescale_compressed: compressed rescaling array from counts to flux space
    :param only_hemisphere: None, "N" (only north) or "S" (only south)
    :return: Processed Fermi counts
    """

    # Extract parameters
    fermi_path = params.gen["fermi_folder"]
    inner_band = params.data["inner_band"]
    outer_rad = params.data["outer_rad"]
    mask_type = params.data["mask_type"]
    remove_exp = params.nn["remove_exp"]

    fermi_data = get_template(params.gen["fermi_folder"], "fermi_map")

    # Mask Galactic plane up to "remove_plane" degrees
    total_mask_neg = make_mask_total(band_mask=inner_band > 0, band_mask_range=inner_band,
                                        mask_ring=outer_rad is not None, inner=0, outer=outer_rad,
                                        nside=hp.npix2nside(len(fermi_data)))
    if mask_type == "3FGL":
        total_mask_neg = (
                    1 - (1 - total_mask_neg) * (1 - get_template(fermi_path, "3FGL_mask")).astype(bool)).astype(bool)
    elif mask_type == "4FGL":
        total_mask_neg = (
                    1 - (1 - total_mask_neg) * (1 - get_template(fermi_path, "4FGL_mask")).astype(bool)).astype(bool)

    if only_hemisphere is not None:
        total_mask_pos_n, total_mask_pos_s = split_into_n_and_s(1 - total_mask_neg,
                                                                nside=hp.npix2nside(len(fermi_data)), filename=None)
        total_mask_neg = (1 - total_mask_pos_s) if only_hemisphere == "S" else (1 - total_mask_pos_n)
        total_mask_neg = total_mask_neg.astype(bool)

    fermi_data[total_mask_neg] = 0.0

    # Ring -> nested
    fermi_data = hp.reorder(fermi_data, r2n=True)

    # Reduce to ROI
    fermi_data = fermi_data[indexes_top]

    # Remove exposure correction
    if rescale_compressed is not None and remove_exp:
        fermi_data = fermi_data / rescale_compressed  # rescale (rescale: nest & ROI)

    return fermi_data
