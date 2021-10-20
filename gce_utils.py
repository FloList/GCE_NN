"""
This file contains useful functions for the GCE NN.
"""
import numpy as np
import scipy.integrate as integrate
import healpy as hp
import os
import pickle
from collections import defaultdict
import errno
import psutil
from shutil import copyfile
import time
import gc
import ray
from numba import jit
from NPTFit import create_mask as cm
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import seaborn as sns
import colorcet as cc
from scipy.stats import norm, wasserstein_distance
from scipy.integrate import trapz
import copy


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
        raise RuntimeError("Error! No npix or nside provided.")
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


def remove_exposure_correction(data, rescale):
    """
    Remove exposure correction by dividing by rescale.
    :param data: numpy array / tensorflow tensor of photon counts at each pixel
    :param rescale: exposure correction (> 1 (< 1) means more (less) exposed than mean, should average to 1
    :return: exposure corrected photon counts
    """
    return data / rescale


def healpix_to_cart(m, nside, lonra, latra, n_pixels, nest=True):
    """Get a cutout via a cartesian projection. Used for 2DCNN (PROJECT_2D option)."""
    proj = hp.projector.CartesianProj(lonra=lonra, latra=latra, coord='G', xsize=n_pixels[0], ysize=n_pixels[1])
    reproj_im = proj.projmap(m, vec2pix_func=lambda x, y, z: hp.vec2pix(nside, x, y, z, nest=nest))  # NOTE: map.sum() != reproj_im.sum() !!!
    return reproj_im


# NOTE: Everything assumed to be NESTED!
def get_pixels(mask, nsides):
    """
    Find pixels in ROI (described by mask) for all values of nsides, such that the entire ROI is contained within the
    output pixels for all the values of nsides.
    For example, if all the unmasked values are contained within the inner pixel at nside = 1, this function will return
    all the pixels at each nside that lie within the coarse nside=1 pixel around the GC.
    :param mask: array that is 1 for all pixels that shall be masked, 0 else
    :param nsides: list of nsides, one entry for each convolutional layer
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
    """
    pixels = [None] * len(nsides)
    pixels[0] = np.argwhere(1 - mask).flatten()
    for i in range(1, len(nsides)):
        mask = hp.ud_grade(1.0 * mask, nside_out=nsides[i], order_in='NEST', order_out='NEST')
        mask[mask < 1] = 0
        pixels[i] = np.argwhere(1 - mask).flatten()
    return pixels


def calc_loglikelihood(pred, true, covar):
    """This function calculates the loglikelihood (modulo a constant factor)."""
    pred, true, covar = np.asarray(pred), np.asarray(true), np.asarray(covar)
    assert pred.shape == true.shape, "Wrong dimensions!"
    if pred.shape[0] != covar.shape[0]:
        try:
            covar = np.reshape(covar, [pred.shape[0], pred.shape[0]])
        except:
            raise ValueError("Covariance matrix has wrong dimensions!")
    assert pred.shape[0] == covar.shape[0] == covar.shape[1], "Covariance matrix has wrong dimensions!"
    term1 = ((true - pred) * (np.linalg.inv(covar) @ (true - pred))).sum()
    term2 = np.log(np.linalg.det(covar))
    return 0.5 * (term1 + term2)


def print_stats(pred, true, models):
    """This function prints a table with the error statistics computed from pred and true."""
    dash = '-' * 68

    mean_error = 100 * np.mean(np.abs(pred - true), 0)
    median_error = 100 * np.median(np.abs(pred - true), 0)
    q95_error = 100 * np.quantile(np.abs(pred - true), .95, 0)
    q99_error = 100 * np.quantile(np.abs(pred - true), .99, 0)

    rms_error = 100 * np.sqrt(((pred - true) ** 2).mean())

    print(dash)
    print('{:<20s}{:>10s}{:>12s}{:>12s}{:>12s}'.format("Errors in %", "Mean", "Median", "95%", "99%"))
    print(dash)
    for i in range(len(mean_error)):
        print('{:<20s}{:>10.2f}{:>12.2f}{:12.2f}{:>12.2f}'.format(models[i], mean_error[i], median_error[i], q95_error[i], q99_error[i]))
    print(dash[:10])
    print("{:<20}{:>10.2f}".format("RMS error:", rms_error))
    print(dash)
    print("calculated based on {:g}".format(pred.shape[0]), "samples.")


def dict_to_array(dic):
    """Convert a dictionary temporarily to an array to make it suitable for the TF pipeline."""
    flatten = lambda *n: (e for a in n
                          for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    return np.asarray(list(flatten([np.asarray(dic[key]).tolist() for key in [*dic]])))


def generate_gce_template(exp, gamma=1.0, r_s=20, d_GC=8.5, nside=128, num_cpus=4, filename="GCE_template"):
    """Generate a template for a generalised NFW profile, everything in kpc"""
    npix = hp.nside2npix(nside)
    # Generalised NFW profile
    nfw = lambda r: 1.0 / ((r / r_s) ** gamma * (1 + r / r_s) ** (3 - gamma))

    # Geometry: if apex = 0: distance
    r = lambda apex_ang, l: np.sqrt((np.sin(apex_ang) * l) ** 2 + (np.cos(apex_ang) * l - d_GC) ** 2)

    # Function to integrate squared density along LOS for Ray
    ray.init(num_cpus=num_cpus)
    print("Running on", num_cpus, "CPUs.")

    @ray.remote
    def nfw_2_int(apex_ang):
        return integrate.quad(lambda l: nfw(r(apex_ang, l)) ** 2.0, 0, np.infty)[0]

    # Define angles (cos(angle) = vec * [1, 0, 0] / (norm(vec) * norm([1, 0, 0]) = vec[0]
    # -> angle = arccos(vec[0]), where [1, 0, 0] is the LOS direction
    angles = np.arccos(hp.pix2vec(nside, range(npix), nest=False)[0])

    # Generate template
    template_raw = np.asarray(ray.get([nfw_2_int.remote(angle) for angle in angles]))

    # Apply exposure correction
    mean_exp = np.mean(exp)
    rescale = exp / mean_exp
    template_corr = template_raw * rescale

    # Normalise template such that mean is 1 in 30* region around GCE (with 2* N/S of Gal. plane masked)
    total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=30)
    template_dmy = template_corr.copy()
    template_dmy[total_mask_neg] = 0
    norm_const = template_dmy[np.argwhere(1 - total_mask_neg)].mean()
    template_final = template_corr / norm_const

    # Save the template
    np.save(filename + "_gamma_" + str(gamma), template_final)

    return template_final


def generate_disk_template(exp, r_s=5, z_s=0.3, d_GC=8.5, nside=128, num_cpus=4, filename="disk_template"):
    """Generate a template for a thin or thick disk, everything in kpc"""
    npix = hp.nside2npix(nside)

    # Disk profile as a function of r and z
    disk = lambda r, z: np.exp(-r / r_s) * np.exp(-np.abs(z) / z_s)

    # Geometry (input is cart. for simplicity)
    z = lambda z_cart, l: l * z_cart
    r = lambda x_cart, y_cart, l: np.sqrt((d_GC - l * x_cart) ** 2 + (l * y_cart) ** 2)
    # r_tot = lambda x_cart, y_cart, z_cart, l: np.sqrt(r(x_cart, y_cart, l) ** 2 + z(z_cart, l) ** 2)
    # r_spher = lambda apex_ang, l: np.sqrt((np.sin(apex_ang) * l) ** 2 + (np.cos(apex_ang) * l - d_GC) ** 2)

    # Function to integrate squared density along LOS for Ray
    ray.init(num_cpus=num_cpus)
    print("Running on", num_cpus, "CPUs.")

    @ray.remote
    def disk_int(x_cart, y_cart, z_cart):
        return integrate.quad(lambda l: disk(r(x_cart, y_cart, l), z(z_cart, l)), 0, np.infty)[0]

    x_, y_, z_ = hp.pix2vec(nside, range(npix), nest=False)

    # Generate template
    template_raw = np.asarray(ray.get([disk_int.remote(x_c, y_c, z_c) for x_c, y_c, z_c in zip(x_, y_, z_)]))

    # Apply exposure correction
    mean_exp = np.mean(exp)
    rescale = exp / mean_exp
    template_corr = template_raw * rescale

    # Normalise template such that sum is 1 in 30* region around GCE (with 2* N/S of Gal. plane masked)
    total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=30)
    template_dmy = template_corr.copy()
    template_dmy[total_mask_neg] = 0
    norm_const = template_dmy[np.argwhere(1 - total_mask_neg)].mean()
    template_final = template_corr / norm_const

    # Save the template
    np.save(filename + "_r_s_" + str(r_s) + "_z_s_" + str(z_s), template_final)

    return template_final


def split_into_N_and_S(template, nside=128, filename="template"):
    """Split up a template into two templates consisting of the N / S hemispheres only. NOTE: RING format!"""
    npix = hp.nside2npix(nside)
    pix_N = np.argwhere(hp.pix2vec(nside, range(npix))[-1] >= 0).flatten()
    pix_S = np.argwhere(hp.pix2vec(nside, range(npix))[-1] < 0).flatten()
    template_N, template_S = np.copy(template), np.copy(template)
    template_N[pix_S] = 0
    template_S[pix_N] = 0
    assert np.all(template_S + template_N == template), "Something went wrong! Aborting..."
    if filename is not None:
        np.save(filename + "_N", template_N)
        np.save(filename + "_S", template_S)
    return template_N, template_S


def mkdir_p(path):
    """Create directory and ignore errors if it exists already."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_is_HPC():
    user_full = os.path.expanduser("~")
    if "flo" in user_full:  # laptop
        HPC = 0
    elif "fl9575" in user_full:  # Gadi: folder should be /home/562/fl9575
        HPC = 1
    elif "flis0155" in user_full:  # Artemis: folder should be /home/flis0155
        HPC = 2
    else:
        raise NotImplementedError("Unknown computing environment! Aborting...")
    return HPC


def get_fermi_folder_basename(HPC, w573=True):
    if HPC == 0:
        if w573:
            fermi_path = '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data_573w'
        else:
            fermi_path = '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'
    elif HPC == 1:
        if w573:
            fermi_path = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data_573w'
        else:
            fermi_path = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data'
    elif HPC == 2:
        if w573:
            fermi_path = '/project/dss/GCE_NN/data/Fermi_data/fermi_data_573w'
        else:
            raise NotImplementedError
    return fermi_path


def backup_train_files(file_in, location, NN_TYPE, param_file=None):
    """This function copies all the relevant files for training to the specified location."""
    files = [file_in]
    folder_in, _ = os.path.split(file_in)
    files += [os.path.join(folder_in, "deepsphere", "models.py")]
    files += [os.path.join(folder_in, "deepsphere", "utils.py")]
    files += [os.path.join(folder_in, "gce_utils.py")]
    files += [os.path.join(folder_in, "gce_data.py")]
    if param_file is None:
        if NN_TYPE == "CNN":
            files += [os.path.join(folder_in, "parameters_CNN.py")]
        elif NN_TYPE == "U-Net":
            files += [os.path.join(folder_in, "parameters_UNet.py")]
    else:
        if param_file.endswith(".py.py"):
            param_file = param_file[:-3]
        files += [os.path.join(folder_in, param_file)]

    for file in files:
        backup_one_file(file, location)


def backup_one_file(file_in, location):
    """This function copies the file file_in to the specified location, with a timestamp added."""
    datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
    file_backup = os.path.join(location, os.path.split(file_in)[-1][:-3] + "_" + datetime + ".py")
    copyfile(file_in, file_backup)


def load_params_from_pickle(folder):
    """This function loads the NN parameters from a 'params' pickle file"""
    all_files = os.listdir(folder)
    params_files = [file for file in all_files if "params" in file and "pickle" in file]
    if len(params_files) > 1:
        print("WARNING! More than one 'params' file was found! Choosing the one that was modified mostly recently!")
        md_times = [os.path.getmtime(os.path.join(folder, file)) for file in params_files]
        params_file = params_files[np.argmax(md_times)]
    else:
        params_file = params_files[0]
    return pickle.load(open(os.path.join(folder, params_file), 'rb'))


def import_from(module, name):
    """from 'module' import 'name'"""
    if module.endswith(".py"):
        module = module[:-3]
    if "/" in module:
        import sys
        path_split = os.path.split(module)
        module = path_split[-1]
        sys.path.append(os.path.join(*path_split[:-1]))
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def get_default_param_file_and_fun(nn_type, pre_gen):
    """get the default parameter file for the given options"""
    if nn_type == "U-Net":
        par_fun_name = "get_params_UNet"
        parameter_filename = "parameters_UNet"
    elif nn_type == "CNN":
        if pre_gen:
            par_fun_name = "get_params_CNN_pre_gen"
            parameter_filename = "parameters_CNN_pre_gen"
        else:
            par_fun_name = "get_params_CNN"
            parameter_filename = "parameters_CNN"
    else:
        raise NotImplementedError("This NN architecture is not defined!")
    return parameter_filename, par_fun_name


def auto_garbage_collect(pct=80.0):
    """This function collects the garbage when the memory consumption is higher than pct percent."""
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


def get_template(fermi_folder, temp):
    """Returns a template."""
    if temp == "iso":
        T = np.load(os.path.join(fermi_folder, 'template_iso.npy'))
    elif temp == "dif":
        T = np.load(os.path.join(fermi_folder, 'template_dif.npy'))
    elif temp == "bub":
        T = np.load(os.path.join(fermi_folder, 'template_bub.npy'))
    elif temp == "bub_var":
        try:
            T = np.load(os.path.join(fermi_folder, 'template_nbub.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_bub_alt.npy'))
    elif temp == "gce":
        try:
            T = np.load(os.path.join(fermi_folder, 'template_gce.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_nfw_g1p0.npy'))
    elif temp == "gce_12":
        try:
            T = np.load(os.path.join(fermi_folder, 'template_gce_gamma_1.20.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_nfw_g1p2.npy'))
    elif temp == "gce_12_N":
        T = np.load(os.path.join(fermi_folder, 'template_gce_gamma_1.20_N.npy'))
    elif temp == "gce_12_S":
        T = np.load(os.path.join(fermi_folder, 'template_gce_gamma_1.20_S.npy'))
    elif temp == "disk":
        T = np.load(os.path.join(fermi_folder, 'template_dsk.npy'))
    elif temp == "thin_disk":
        try:
            T = np.load(os.path.join(fermi_folder, 'template_disk_r_s_5_z_s_0.3.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_dsk_z0p3.npy'))
    elif temp == "thick_disk":
        try:
            T = np.load(os.path.join(fermi_folder, 'template_disk_r_s_5_z_s_1.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_dsk_z1p0.npy'))
    elif temp == "dif_O_pibs":
        try:
            T = np.load(os.path.join(fermi_folder, 'ModelO_r25_q1_pibrem.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_Opi.npy'))
    elif temp == "dif_O_ic":
        try:
            T = np.load(os.path.join(fermi_folder, 'ModelO_r25_q1_ics.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_Oic.npy'))
    elif temp == "dif_A_pibs":
        T = np.load(os.path.join(fermi_folder, 'template_Api.npy'))
    elif temp == "dif_A_ic":
        T = np.load(os.path.join(fermi_folder, 'template_Aic.npy'))
    elif temp == "dif_F_pibs":
        T = np.load(os.path.join(fermi_folder, 'template_Fpi.npy'))
    elif temp == "dif_F_ic":
        T = np.load(os.path.join(fermi_folder, 'template_Fic.npy'))
    elif temp == "psc_3":
        try:
            T = np.load(os.path.join(fermi_folder, 'template_psc.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'template_psc_3fgl.npy'))
    elif temp == "psc_4":
        T = np.load(os.path.join(fermi_folder, 'template_psc_4fgl.npy'))
    elif temp == "3FGL_mask":
        try:
            T = np.load(os.path.join(fermi_folder, 'fermidata_pscmask.npy'))
        except FileNotFoundError:
            T = np.load(os.path.join(fermi_folder, 'fermidata_pscmask_3fgl.npy'))
    elif temp == "4FGL_mask":
        T = np.load(os.path.join(fermi_folder, 'fermidata_pscmask_4fgl.npy'))
    elif temp == "exp":
        T = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
    elif temp == "counts":
        T = np.load(os.path.join(fermi_folder, "fermidata_counts.npy"))
    else:
        raise NotImplementedError("Template", temp, "not available!")
    return T


def dnds(theta, s, a_is_log=False):
    """
    dN/dS values for NPT corresponding to model parameters theta for the array of counts s. This function is adapted
    from the NPTF implementation of dnds.
    NOTE: the template normalisation in theta is expected to be log10(A).
    """

    nbreak = int((len(theta) - 2)/2.)

    # Get APS (float) and slopes/breaks (arrays)
    a_ps, n_ary, sb_ary = theta[0], theta[1:nbreak+2], \
                          theta[nbreak+2:]

    if a_is_log:
        a_ps = 10 ** a_ps

    # Determine where the s values fall with respect to the breaks
    where_vecs = [[] for _ in range(nbreak+1)]
    where_vecs[0] = np.where(s >= sb_ary[0])[0]
    for i in range(1, nbreak):
        where_vecs[i] = np.where((s >= sb_ary[i]) & (s < sb_ary[i-1]))[0]
    where_vecs[-1] = np.where(s < sb_ary[-1])[0]

    # Calculate dnds values for a broken power law with arbitrary breaks
    dnds = np.zeros(len(s))
    dnds[where_vecs[0]] = a_ps*(s[where_vecs[0]]/sb_ary[0])**(-n_ary[0])
    dnds[where_vecs[1]] = a_ps*(s[where_vecs[1]]/sb_ary[0])**(-n_ary[1])

    for i in range(2, nbreak+1):
        dnds[where_vecs[i]] = \
            10**a_ps*np.prod([(sb_ary[j+1]/sb_ary[j])**(-n_ary[j+1])
                              for j in range(0, i-1)]) * \
            (s[where_vecs[i]]/sb_ary[i-1])**(-n_ary[i])

    return dnds


def return_intensity_arrays_poiss(A, temp_map, exp_masked, counts=False, nside=128, a_is_log=False):
    """ Return intensity / count array of a Poissonian template (adapted from NPTFit)
        :param A: log10 of the template normalisation
        :param temp_map: masked template map
        :param exp_masked: masked exposure map
        :param counts: whether to return counts (or intensities, by default)
        :param nside: HEALPix nside
        :param a_is_log: is A provided or log10(A)?
    """
    # If intensity, convert from counts to counts/cm^2/s/sr
    pixarea = hp.nside2pixarea(nside)
    if counts:
        template_sum = np.nansum(temp_map)
    else:
        template_sum = np.nanmean(temp_map / exp_masked / pixarea)

    # Get PT intensities by scaling the compressed mask intensity
    # by the relevant normalizations from chains
    if a_is_log:
        A = 10 ** A
    intensity_array_poiss = template_sum * A

    return intensity_array_poiss


def return_intensity_arrays_non_poiss(theta, temp_map, exp_masked, smin=0.01, smax=10000,
                                      nsteps=10000, counts=False, nside=128,
                                      a_is_log=False):
    """ Return intensity / count array of a non-Poissonian template (adapted from NPTFit)
        :param theta: model parameters
        :param temp_map: masked non-Poissonian template map (that is, masked pixels are 0)
        :param exp_masked: masked exposure map (that is, masked pixels have exposure = 0)
        :param smin: minimum count to "integrate" dnds from
        :param smax: maximum count to "integrate" dnds to
        :param nsteps: number of count bins in sum approximation of integral
        :param counts: whether to return counts (or intensities, by default)
        :param nside: HEALPix nside
        :param a_is_log: is A provided or log10(A)?
    """
    pixarea = hp.nside2pixarea(nside)
    # If intensity, convert from counts to counts/cm^2/s/sr
    if counts:
        template_sum = np.nansum(temp_map)
    else:
        template_sum = np.nanmean(temp_map / exp_masked / pixarea)

    sarray = 10**np.linspace(np.log10(smin), np.log10(smax), nsteps)
    ds = [sarray[i+1]-sarray[i] for i in range(len(sarray)-1)]
    ds = np.array(ds + [ds[-1]])

    # Get NPT intensity arrays. These are calculated as
    # \int(dS*S*dN/dS). Note that the A^{PS} parameter is a
    # rescaling of the counts, which is why to get the
    # intensity this is multiplied by the total counts
    intensity_array_non_poiss = np.sum(template_sum * dnds(theta, sarray, a_is_log=a_is_log) * sarray * ds)
    return intensity_array_non_poiss


def flip_map(map, hor=False, vert=False):
    """Flip a healpix map horizontally / vertically"""
    map = np.copy(map) # need to make a copy in order to not overwrite the input
    if len(map.shape) == 1:
        map = np.expand_dims(map, 1)
    nside = hp.get_nside(map[:, 0])
    npix = hp.nside2npix(nside)
    if vert:
        theta_phi = np.vstack(hp.pix2ang(nside, np.arange(npix))).T
        theta_phi_v_flip = theta_phi
        theta_phi_v_flip[:, 0] = np.pi - theta_phi[:, 0]
        pix_v_flip = hp.ang2pix(nside, theta_phi_v_flip[:, 0], theta_phi_v_flip[:, 1])
        map = map[pix_v_flip, :]
    if hor:
        theta_phi = np.vstack(hp.pix2ang(nside, np.arange(npix))).T
        theta_phi_h_flip = theta_phi
        theta_phi_h_flip[:, 1] = 2 * np.pi - theta_phi[:, 1]
        pix_h_flip = hp.ang2pix(nside, theta_phi_h_flip[:, 0], theta_phi_h_flip[:, 1])
        map = map[pix_h_flip, :]
    if map.shape[1] == 1:
        map = np.squeeze(map, 1)
    return map


@jit(nopython=True)
def find_last_pos(vec):
    """return the index of the last positive entry in vec, -1 if none is found"""
    for i in range(len(vec)-1, -1, -1):
        if vec[i] > 0:
            return i
    return -1


def get_max_rad_with_counts(maps, unmasked_pix, nside=128, nest=True):
    """
    This function returns the maximum angle w.r.t. the GC within which each maps contains counts.
    NOTE: by default, NESTED format is assumed!
    """
    theta, phi = hp.pix2ang(nside, unmasked_pix, nest=nest, lonlat=True)
    ang_dist_from_GC = hp.rotator.angdist([theta, phi], [0, 0], lonlat=True)
    ang_dist_from_GC *= 180.0 / np.pi
    sorted_dist_ind = np.argsort(ang_dist_from_GC.flatten())
    maps_sorted = maps[:, sorted_dist_ind]
    largest_ind = np.apply_along_axis(find_last_pos, 1, maps_sorted)
    max_rad_with_counts = ang_dist_from_GC[sorted_dist_ind[largest_ind]]
    return max_rad_with_counts


def Fermi_PSF(r):
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

    def Fermi_PSF_inner(r):
        return fcore * king_fn(r / spe, score, gcore) + (1 - fcore) * king_fn(r / spe, stail, gtail)

    return Fermi_PSF_inner(r)


def get_Fermi_PDF_sampler(n_points_f=int(1e6)):
    f = np.linspace(0, np.pi, n_points_f)
    pdf_psf = f * Fermi_PSF(f)
    pdf = PDFSampler(f, pdf_psf)
    return pdf


def fibonacci(n):
    """
    This function returns the n-th Fibonacci number.
    :param n: index
    :return: n-th Fibonacci number
    """
    if n in [0, 1]:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def get_filtered_data(folder_in, cond, n_max=None, filename_out="filtered_data"):
    """
    This function can be used to get a filtered data set
    :param folder_in: input folder
    :param cond: condition that needs to be fulfilled as a lambda function, e.g.:
                 lambda d: np.logical_and(0.05 <= d['flux_fraction']['gce_12'] + d['flux_fraction']['gce_12_PS'],
                                          d['flux_fraction']['gce_12'] + d['flux_fraction']['gce_12_PS'] <= 0.30)
                 conditions on d['data'] are possible, too.
    :param n_max: max. numbers of samples
    :param filename_out: output filename
    :return filtered data dictionary
    """
    folder_content = os.listdir(folder_in)
    folder_content = [file for file in folder_content if file.endswith(".pickle")]  # only take pickle files
    n_found = 0
    filtered_data_dict = dict()
    filtered_data_dict["data"] = []
    filtered_data_dict["flux_fraction"] = dict()
    filtered_data_dict["info"] = dict()

    assert folder_content, "No suitable data found!"

    for i_file, file in enumerate(folder_content):
        data_file = open(os.path.join(folder_in, file), 'rb')
        data_dict = pickle.load(data_file)
        data_file.close()
        cond_true = cond(data_dict)
        if cond_true.sum() + n_found > (n_max or np.infty):
            new_allowed = n_max - n_found
            cond_true[np.argwhere(cond_true)[new_allowed][0]:] = False

        # Get data
        filtered_data_dict["data"].append(data_dict["data"][:, cond_true])

        # Get flux fractions
        all_models = data_dict["flux_fraction"].keys()
        for model in all_models:
            if not model in filtered_data_dict["flux_fraction"].keys():
                filtered_data_dict["flux_fraction"][model] = []
            filtered_data_dict["flux_fraction"][model].append(data_dict["flux_fraction"][model][cond_true])

        # Get info
        for key in data_dict["info"].keys():
            if key not in filtered_data_dict["info"].keys():
                filtered_data_dict["info"][key] = dict()
            for model in all_models:
                if model in data_dict["info"][key].keys():
                    if model not in filtered_data_dict["info"][key].keys():
                        filtered_data_dict["info"][key][model] = []
                    filtered_data_dict["info"][key][model].append(data_dict["info"][key][model][cond_true])

        n_found += cond_true.sum()
        if n_found >= (n_max or np.infty):
            break

    filtered_data_dict["data"] = np.column_stack(filtered_data_dict["data"])
    for model in all_models:
        filtered_data_dict["flux_fraction"][model] = np.concatenate(filtered_data_dict["flux_fraction"][model])
        for key in data_dict["info"].keys():
            if model in filtered_data_dict["info"][key].keys():
                filtered_data_dict["info"][key][model] = np.concatenate(filtered_data_dict["info"][key][model])

    with open(os.path.join(folder_in, filename_out + ".pickle"), 'wb') as f:
        pickle.dump(filtered_data_dict, f)
        print("Filtered data file written.")

    return filtered_data_dict


# TODO: so far: does NOT incorporate non-diagonal elements of covariance matrix!
def calculate_coverage(pred, covar, true, levels=[1, 2, 3]):
    """
    Calculate fraction of samples within the given confidence intervals (levels in sigma), assuming Gaussianity.
    This is justified if the aleatoric uncertainties are modelled as Gaussians and the epistemic uncertainties are small
    :param pred: predicted means: n_MC x n_batch x n_templates
    :param covar: predicted ALEAT.(!) covariance matrix: n_MC x n_batch x n_templates x n_templates
                  (epistemic uncertainty is computed within this function from the MC samples)
    :param true: true flux fractions: n_batch x n_templates
    :param levels: uncertainty levels in sigma
    """

    # if length is 2: assume there is no MC dropout sampling
    if len(pred.shape) == 2:
        pred = pred[None]
        covar = covar[None]

    n_MC, n_samples, n_params = pred.shape

    # Calculate uncertainties (assuming diagonal covariance matrices)
    var_aleatoric = (np.asarray([[np.diag(covar[j, i]) for i in range(n_samples)] for j in range(n_MC)])).mean(0)
    var_epistemic = pred.var(0)  # calculate from true samples, NOT from perturbed samples
    var_predictive = var_aleatoric + var_epistemic

    within_interv = np.zeros((len(levels), n_samples, n_params))
    for i_lev, conf_lev in enumerate(levels):
        within_interv[i_lev] = np.logical_and((true <= pred.mean(0) + conf_lev * np.sqrt(var_predictive)),
                                              (true >= pred.mean(0) - conf_lev * np.sqrt(var_predictive)))

    coverage = within_interv.sum(1) / n_samples
    return coverage


def make_error_plot(MODELS, real_fluxes, pred_fluxes, colours=None, NPTFit_fluxes=None, delta_y=0, model_names=None,
                    out_file="error_plot.pdf", legend=None, show_stripes=True, show_stats=True, marker="^",
                    marker_NPTF="o", ms=40, ms_NPTFit=16, alpha=0.4, lw=0.8, lw_NPTFit=1.6, text_colours=None,
                    cmap="magma", pred_covar=None, ecolor=None, ticks=None, vmin=None, vmax=None, show_ticks=False):
    """
    Make an error plot of the NN predictions
    :param MODELS: models to plot
    :param real_fluxes: true flux fractions
    :param pred_fluxes: NN estimates of the flux_fractions
    :param colours: colours to use for plotting (default: settings from GCE letter)
    :param NPTFit_fluxes: if provided: also plot NPTFit fluxes
    :param delta_y: shift vertical position of the stats
    :param model_names: names of the models, defaults to MODELS
    :param out_file: name of the output file
    :param legend: show legend? by default on if NN and NPTFit fluxes are given, otherwise off
    :param show_stripes: show the stripes indicating 5% and 10% errors?
    :param show_stats: show stats?
    :param marker: marker for NN estimates
    :param marker_NPTF: marker for NPTFit estimates
    :param ms: marker size for NN estimates
    :param ms_NPTFit: marker size for NPTFit estimates
    :param alpha: alpha for the markers
    :param lw: linewidth for the markers
    :param lw_NPTFit: linewidth for the NPTFit markers
    :param text_colours: defaults to colours
    :param cmap: default: "magma"
    :param pred_covar: predicted covariances for an error bar plot (non-diag. elements are ignored)
    :param ecolor: errorbar colour
    :param ticks: ticks
    :param vmin / vmax: limits for colourmap
    :param show_ticks: show ticks?
    :return figure and axes
    """
    if colours is None:
        colours = ['#ff0000', '#ec7014', '#fec400', '#37c837', 'deepskyblue', 'darkslateblue', 'k']
    sns.set_style("ticks" if show_ticks else "white")
    sns.set_context("talk")
    if legend is None:
        legend = True if NPTFit_fluxes is not None else False
    if model_names is None:
        model_names = MODELS
    n_col = max(int(np.ceil(np.sqrt(len(MODELS)))), 1)
    n_row = int(np.ceil(len(MODELS) / n_col))
    if ticks is None:
        ticks = [0, 0.2, 0.4, 0.6, 0.8]
    x_ticks = y_ticks = ticks

    if text_colours is None and len(colours[0]) == 1:
        text_colours = colours
    elif text_colours is None:
        text_colours = ["k"] * len(MODELS)

    # Calculate errors
    mean_abs_error = np.mean(np.abs(pred_fluxes - real_fluxes), 0)
    max_abs_error = np.max(np.abs(pred_fluxes - real_fluxes), 0)
    # q95_abs_error = np.quantile(np.abs(pred_fluxes - real_fluxes), .95, axis=0)
    # q99_abs_error = np.quantile(np.abs(pred_fluxes - real_fluxes), .99, axis=0)
    if NPTFit_fluxes is not None:
        mean_abs_error_NP = np.mean(np.abs(NPTFit_fluxes - real_fluxes), 0)
        max_abs_error_NP = np.max(np.abs(NPTFit_fluxes - real_fluxes), 0)
        # q95_abs_error_NP = np.quantile(np.abs(NPTFit_fluxes - real_fluxes), .95, axis=0)
        # q99_abs_error_NP = np.quantile(np.abs(NPTFit_fluxes - real_fluxes), .99, axis=0)

    scat_fig, scat_ax = plt.subplots(n_row, n_col, figsize=(11.64, 8), squeeze=False, sharex="all", sharey="all")
    for i_ax, ax in enumerate(scat_ax.flatten()):
        if i_ax >= len(MODELS):
            continue
        ax.plot([0, 1], [0, 1], 'k-', lw=2, alpha=0.5)
        if show_stripes:
            ax.fill_between([0, 1], y1=[0.05, 1.05], y2=[-0.05, 0.95], color="0.7", alpha=0.5)
            ax.fill_between([0, 1], y1=[0.1, 1.1], y2=[-0.1, 0.9], color="0.9", alpha=0.5)
        ax.set_aspect("equal", "box")

    for i_ax, ax in enumerate(scat_ax.flatten(), start=0):
        if i_ax >= len(MODELS):
            ax.axis("off")
            continue
        if NPTFit_fluxes is not None:
            ax.scatter(real_fluxes[:, i_ax], NPTFit_fluxes[:, i_ax], s=ms_NPTFit, c="1.0", marker=marker_NPTF, lw=lw_NPTFit,
                       alpha=alpha, edgecolor="k", zorder=2, label="NPTFit", cmap=cmap, vmin=vmin, vmax=vmax)
        if pred_covar is None:
            ax.scatter(real_fluxes[:, i_ax], pred_fluxes[:, i_ax], s=ms, c=colours[i_ax], marker=marker,
                       lw=lw, alpha=alpha, edgecolor="k", zorder=3, label="NN", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            if len(np.asarray(colours).shape) > 1:
                ax.errorbar(x=real_fluxes[:, i_ax], y=pred_fluxes[:, i_ax], fmt='none',
                            alpha=alpha, zorder=3, label="",
                            yerr=np.sqrt(pred_covar[:, i_ax, i_ax]), elinewidth=2)
                ax.scatter(real_fluxes[:, i_ax], pred_fluxes[:, i_ax], s=ms, c=colours[i_ax], marker=marker,
                           lw=lw, alpha=alpha, edgecolor="k", zorder=3, label="NN", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.errorbar(x=real_fluxes[:, i_ax], y=pred_fluxes[:, i_ax], fmt=marker, ms=ms, mfc=colours[i_ax], mec="k",
                            ecolor=ecolor or colours[i_ax], lw=lw, alpha=alpha, zorder=3, label="NN", yerr=np.sqrt(pred_covar[:, i_ax, i_ax]),
                            elinewidth=2)
        if i_ax == 0 and legend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], frameon=True, loc='upper left', bbox_to_anchor=(0, 0.85),
                      handletextpad=0.07, borderpad=0.25, fontsize=14)
        if show_stats:
            ax.text(0.65, 0.24 + delta_y, r"$\it{Mean}$", ha="center", va="center", size=12)
            ax.text(0.65, 0.17 + delta_y, "{:.2f}%".format(mean_abs_error[i_ax] * 100), ha="center", va="center",
                    color=text_colours[i_ax], size=12)
            if NPTFit_fluxes is not None:
                ax.text(0.65, 0.10, "{:.2f}%".format(mean_abs_error_NP[i_ax] * 100), ha="center", va="center", size=12)
            ax.text(0.9, 0.24 + delta_y, r"$\it{Max}$", ha="center", va="center", size=12)
            ax.text(0.9, 0.17 + delta_y, "{:.2f}%".format(max_abs_error[i_ax] * 100), ha="center", va="center",
                    color=text_colours[i_ax], size=12)
            if NPTFit_fluxes is not None:
                ax.text(0.9, 0.10, "{:.2f}%".format(max_abs_error_NP[i_ax] * 100), ha="center", va="center", size=12)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        if i_ax >= n_col * (n_row - 1):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
        if i_ax % n_col == 0:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks)
        ax.text(0.03, 0.97, model_names[i_ax], va="top", ha="left")

    plt.tight_layout()
    plt.subplots_adjust(top=0.91, bottom=0.105, left=0.085, right=0.915, hspace=0.0, wspace=0.0)
    scat_fig.text(0.5, 0.025, "True", ha="center", va="center")
    scat_fig.text(0.02, 0.5, "Estimated", ha="center", va="center", rotation="vertical")
    pretty_plots()
    plt.show()

    if out_file is not None:
        scat_fig.savefig(out_file, bbox_inches="tight")

    return scat_fig, scat_ax


def multipage(filename, figs=None, dpi=360):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        # RASTERIZE???
        # fig.set_rasterized(True)
        fig.savefig(pp, format='pdf', dpi=dpi)
    pp.close()


def plot_some_examples(params, MODELS, test_data, real_fluxes, pred_fluxes, multiply_by=None, PROJECT_2D=False, start_ind=0,
                       n_examples_per_dim=3, marg=0.04, only_maps=False, figsize=(16, 12), nest=True,
                       zoom_xlim=None, zoom_ylim=None, cbar=True, fermi_folder=None, mark_3FGL=False,
                       col_3FGL="gray", cmap="rocket"):
    """
    :param params: params dictionary
    :param MODELS: model strings
    :param test_data: data for plotting
    :param real_fluxes: true flux fractions
    :param pred_fluxes: predicted flux fractions
    :param multiply_by: multiply the map for example by the exposure correction in order to get integer counts
    :param PROJECT_2D: 2DCNN instead of DeepSphere architecture?
    :param start_ind: start index
    :param n_examples_per_dim: number of maps to plot per dimension (x/y)
    :param marg: margin for axes
    :param only_maps: only plot maps without fluxes. This overrides the values for marg, cbar, and figsize.
    :param figsize: figure size
    :param nest: HEALPix nested scheme instead of ring
    :param zoom_xlim: zoom into the map in x-direction (default: [-0.3, 0.3])
    :param zoom_ylim: zoom into the map in y-direction (default: [-0.4, 0.4])
    :param cbar: show colourbar
    :param fermi_folder: folder with templates, needed if 3FGL sources shall be masked
    :param mark_3FGL: mark 3FGL sources (not implemented for unmask / PROJECT_2D)
    :param col_3FGL: colour for 3FGL sources
    :param cmap: default: "rocket"
    :return: fix, axs
    Note: if unmask / PROJECT_2D is on, some options are not implemented!
    """

    assert not mark_3FGL or fermi_folder is not None, "No Fermi folder is specified!"

    if zoom_xlim is None:
        zoom_xlim = [-0.3, 0.3]
    if zoom_ylim is None:
        zoom_ylim = [-0.4, 0.4]

    if only_maps:
        marg = 5e-4
        figsize = (6, 8)
        cbar = False

    fig_ex, axs_ex = plt.subplots(n_examples_per_dim, n_examples_per_dim, figsize=figsize)
    for i_ax, ax in enumerate(axs_ex.flatten()):
        if not only_maps:
            s_ = ', '
            s_r = s_.join([str("{:.2f}".format(np.round(real_fluxes[i_ax + start_ind, i], 2))) for i in range(len(MODELS))])
            s_p = s_.join([str("{:.2f}".format(np.round(pred_fluxes[i_ax + start_ind, i], 2))) for i in range(len(MODELS))])
            title = '\n'.join([s_r] + [s_p])
        else:
            title = ''

        if PROJECT_2D:
            if multiply_by is not None:
                test_data_plot = test_data * multiply_by
            else:
                test_data_plot = test_data
            ax.imshow(test_data_plot[i_ax + start_ind, :], origin="lower")
        else:
            if params["unmask"]:
                if multiply_by is not None:
                    test_data_plot = test_data * multiply_by
                else:
                    test_data_plot = test_data
                hp.mollview(test_data_plot[i_ax + start_ind, :], nest=nest,
                            sub=[n_examples_per_dim, n_examples_per_dim, i_ax + 1],
                            title=title, margins=4 * [marg], cbar=cbar)
            else:
                if multiply_by is not None:
                    test_data_plot = masked_to_full(test_data[i_ax + start_ind, :], params["indexes"][0], fill_value=0.0,
                                                    nside=params["nsides"][0]) * multiply_by
                else:
                    test_data_plot = masked_to_full(test_data[i_ax + start_ind, :], params["indexes"][0], fill_value=0.0,
                                                    nside=params["nsides"][0])
                if mark_3FGL:
                    assert nest, "This option is only implemented for the nested scheme!"
                    fgl3 = get_template(fermi_folder, "3FGL_mask")
                    total_mask_pos = (1 - cm.make_mask_total(nside=params["nsides"][0], band_mask=True, band_mask_range=2,
                                                             mask_ring=True, inner=0,
                                                             outer=get_max_rad_with_counts(test_data_plot[None], range(hp.nside2npix(params["nsides"][0])), params["nsides"][0])))
                    mask = hp.reorder(fgl3 * total_mask_pos, r2n=True).astype(bool)
                    test_data_plot[mask] = np.nan

                try:
                    hp.mollview(test_data_plot,
                                nest=nest, sub=[n_examples_per_dim, n_examples_per_dim, i_ax + 1],
                                title=title, margins=4 * [marg], cbar=cbar, badcolor=col_3FGL, cmap=cmap)
                except TypeError:  # some healpy version don't have badcolor!
                    hp.mollview(test_data_plot,
                                nest=nest, sub=[n_examples_per_dim, n_examples_per_dim, i_ax + 1],
                                title=title, margins=4 * [marg], cbar=cbar, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.axis("off")
            mw_ax = plt.gca()
            if zoom_xlim is not None:
                mw_ax.set_xlim(zoom_xlim)
            if zoom_ylim is not None:
                mw_ax.set_ylim(zoom_ylim)

    print(MODELS)
    return fig_ex, axs_ex


def get_all_data(folder, downscale_nside=None, original_nside=None, only_with_name=None):
    """
    This function returns all the data found in a folder, together with the settings dictionary
    The "info" field is NOT returned (which contains further information on the SCD parameters etc.
    :param folder: data folder
    :param downscale_nside: downscale data to this resolution
    :param original_nside: original resolution
    :param only_with_name: only select files containing this string
    :return: all_data, all_labels, settings_dict
    """
    all_files = os.listdir(folder)
    all_files = [file for file in all_files if "EXCLUDE" not in file]
    settings_ind = np.argwhere(["settings" in file for file in all_files])[0][0]
    settings_file = open(os.path.join(folder, all_files[settings_ind]), 'rb')
    settings_dict = pickle.load(settings_file)
    settings_file.close()
    all_data = []
    all_labels = defaultdict()
    for i_file, file in enumerate(all_files):
        if i_file == settings_ind:
            pass
        else:
            this_file = os.path.join(folder, file)
            if only_with_name is None or only_with_name in this_file:
                data_file = open(this_file, 'rb')
                data_dict = pickle.load(data_file)
                data_file.close()
                for key in list(data_dict["flux_fraction"].keys()):
                    try:
                        all_labels[key] = np.hstack([all_labels[key], data_dict["flux_fraction"][key]])
                    except KeyError:
                        all_labels[key] = data_dict["flux_fraction"][key]
                if downscale_nside is None:
                    data_add = data_dict["data"]
                else:
                    data_add = masked_to_full(data_dict["data"].T, hp.nside2npix(original_nside), settings_dict["unmasked_pix"]).T
                    data_add = hp.ud_grade(data_add.T, downscale_nside, power=-2).T
                if len(all_data) == 0:
                    all_data = data_add
                else:
                    all_data = np.hstack([all_data, data_add])
    # also downscale "rescale"
    if downscale_nside is not None:
        settings_dict["rescale"] = hp.ud_grade(settings_dict["rescale"], downscale_nside)
    return all_data, all_labels, settings_dict


# Get indices from a list of nside
# OBSOLETE
def get_indices(nsides, unmasked_pix):
    indices = [None] * len(nsides)
    indices[0] = unmasked_pix
    nans = np.ones(hp.nside2npix(nsides[0])) * (-1.6375e+30)  # healpix NaN value = -1.6375e30
    nans[unmasked_pix] = 0.0
    for i_nside, nside in enumerate(nsides[1:], start=1):
        downscaled = hp.ud_grade(nans, nside_out=nside)
        indices[i_nside] = np.argwhere(downscaled != -1.6375e+30).flatten()
    return np.asarray(indices)


# Set everything in serif fonts
def pretty_plots():
    # Get all figures
    # Plot settings
    mpl.rcParams['text.usetex'] = False
    mpl.rc('font', family='serif')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \mathrm command
    #mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'

    ########################################################################################################################
    ########################################################################################################################
    # Define settings
    font_family = 'serif'  # 'sans-serif', 'serif'
    for i in plt.get_fignums():
        gcf = plt.figure(i)

        # Get all axes
        for i_ax in range(len(gcf.axes)):

            # All axis labels in correct font
            gcf.axes[i_ax].xaxis.get_label().set_family(font_family)
            gcf.axes[i_ax].yaxis.get_label().set_family(font_family)

            # All ticklabels in correct font
            xticklabs = gcf.axes[i_ax].xaxis.get_ticklabels()
            for i_lab in range(len(xticklabs)):
                xticklabs[i_lab].set_family(font_family)

            yticklabs = gcf.axes[i_ax].yaxis.get_ticklabels()
            for i_lab in range(len(gcf.axes[i_ax].yaxis.get_ticklabels())):
                yticklabs[i_lab].set_family(font_family)

            # Everything for axes in correct font
            all_text = gcf.axes[i_ax].findobj(match=mpl.text.Text)
            for j in range(len(all_text)):
                all_text[j].set_family(font_family)

        # Everything for figure in correct font
        all_text_fig = gcf.findobj(match=mpl.text.Text)
        for k in range(len(all_text_fig)):
            all_text_fig[k].set_family(font_family)


def combine_list_of_dicts(dict_list):
    """
    This helper function combines a list of n_dict dictionaries to dictionary with n_dim + 1 dimensional tensors,
    where the first dimension corresponds to the dict_list entries.
    :param dict_list: List of dictionaries
    :return: combined tensor
    """
    all_keys = np.unique([list(dict_list[i].keys()) for i in range(len(dict_list))])
    dict_out = {key:[] for key in all_keys}
    for key in all_keys:
        dict_out[key] = np.concatenate(np.expand_dims([dict_list[i][key] for i in range(len(dict_list))], 0))
    return dict_out


def plot_gaussian_mixtures(alphas, means, sigmas, truths=None, n_ffs=5e4, colours=None, plot_components=True,
                           truncated=True):
    """
    This function generates plots of the distributions described by a mixture of n_Gaussian Gaussian.
    :param alphas: array with shape n_maps x n_Gaussians x n_templates: coefficients for the Gaussians
    :param means: array with shape n_maps x n_Gaussians x n_templates: means of the Gaussians
    :param sigmas: array with shape n_maps x n_Gaussians x n_templates: stds of the Gaussians
    :param truths: can be used to plot the truths as lines (shape: n_batch x n_templates)
    :param n_ffs: number of FFs to evaluate PDF on
    :param colours: colours for plotting
    :param plot_components: if True: plot the PDFs of the underlying Gaussians
    :param truncated: are the Gaussians truncated between [0, 1]?
    :returns figures
    """
    alphas, means, sigmas = np.asarray(alphas), np.asarray(means), np.asarray(sigmas)
    assert alphas.shape == means.shape == sigmas.shape, "Input shapes don't match!"
    if alphas.ndim == 2:
        alphas, means, sigmas = np.expand_dims(alphas, 0), np.expand_dims(means, 0), np.expand_dims(sigmas, 0)
    if truths is not None:
        truths = np.asarray(truths)
        if truths.ndim == 1:
            truths = np.expand_dims(truths, 0)
    n_maps, n_Gaussians, n_templates = alphas.shape

    if colours is None:
        colours = ['#ff0000', '#ec7014', '#fec400', '#37c837', 'deepskyblue', 'darkslateblue', 'k']

    # Create array with FFs
    n_ffs = int(n_ffs)
    ff_vec = np.linspace(0, 1, n_ffs)

    # Iterate over the maps
    figs = [None] * n_maps
    for i_map in range(n_maps):
        figs[i_map], ax = plt.subplots(1, 1, figsize=(10, 10))
        this_pdf = [None] * n_templates
        for i_template in range(n_templates):
            all_pdfs = np.asarray([norm.pdf(ff_vec, loc=means[i_map, i_Gauss, i_template], scale=sigmas[i_map, i_Gauss, i_template]) \
                                   for i_Gauss in range(n_Gaussians)]).T
            if truncated:
                all_pdfs /= (norm.cdf((1.0 - means[i_map, :, i_template]) / sigmas[i_map, :, i_template]) - norm.cdf((0.0 - means[i_map, :, i_template]) / sigmas[i_map, :, i_template]))
            all_alphas = alphas[i_map, :, i_template]
            if plot_components:
                ax.plot(ff_vec, all_pdfs * np.expand_dims(all_alphas, 0), lw=2, color=colours[i_template],
                        alpha=1.0, linestyle="--")
            this_pdf[i_template] = all_pdfs @ all_alphas
            ax.fill_between(ff_vec, this_pdf[i_template], lw=2, edgecolor="k", facecolor=colours[i_template], alpha=0.6)
            if truths is not None:
                ax.axvline(truths[i_map, i_template], lw=4, color=colours[i_template], zorder=-5)
        ax.set_title("Map " + str(i_map), fontsize=24)
        ax.set_ylim([0, np.asarray(this_pdf)[:, n_ffs // 100:].max()])
        ax.set_xlabel("Flux fractions", fontsize=24)
        ax.set_ylabel("Probability", fontsize=24)
        pretty_plots()

    return figs


def make_histogram_plot(model, tau_vec, bin_centres, params, test_out, n_plot=1, inner_band=2, name="examples_PDF_CDF.pdf",
                        mean_exp=0, continuous_style=False, width=None, cmap_hp="rocket_r", cmap_hp_res="RdBu_r", map_log=False,
                        additional_histogram=None, temp_names=None, residual_clim_fac=1.0, print_counts=True):
    # Check if histogram label is given
    if "gce_hist" in test_out.keys():
        test_gce_hist = test_out["gce_hist"]
        has_hist_label = True
    else:
        has_hist_label = False

    # Get width of bins
    if width is None:
        width = min(np.diff(bin_centres))

    # Get number of quantile levels
    n_taus = len(tau_vec)
    colors = cc.cm.bkr(np.linspace(0, 1, n_taus))[::-1]

    # Define the indices for the maps to plot
    plot_samples = np.arange(n_plot)

    # Set some plot settings
    cum_col_1 = [0.25490196, 0.71372549, 0.76862745, 1]
    cum_col_faint = [0.25490196, 0.71372549, 0.76862745, 0.2]

    sns.set_context("paper")
    sns.set_style("ticks")
    total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band, mask_ring=True, inner=0,
                                        outer=params["outer_rad"], nside=params["nside"])
    if params["mask_type_fermi"] == "3FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "3FGL_mask"))).astype(bool)
    elif params["mask_type_fermi"] == "4FGL":
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(params["template_path"], "4FGL_mask"))).astype(bool)
    total_mask_neg = hp.reorder(total_mask_neg, r2n=True)

    for i_sample, db_sample in enumerate(plot_samples):
        test_out_loc = copy.copy(test_out)

        # Tile for the different quantile levels
        for key in test_out.keys():
            test_out_loc[key] = np.tile(test_out[key][db_sample, :][None], [n_taus] + [1] * (len(test_out[key].shape) - 1))

        # Predict and get means and histograms
        pred_fluxes_dict = model.predict(test_out_loc, None, tau_hist=np.expand_dims(tau_vec, -1))
        pred_fluxes = pred_fluxes_dict["logits_mean"]
        pred_hist = pred_fluxes_dict["gce_hist"]
        n_channels = pred_hist.shape[2]
        if temp_names is None:
            temp_names = ["Template " + str(i) for i in range(1, 1 + n_channels)]
        has_alea = "covar" in pred_fluxes_dict.keys()

        # Is residual map computed?
        if np.all(np.isin(["count_maps_modelled_Poiss", "count_maps_residual"], list(pred_fluxes_dict.keys()))):
            plot_res = True
        else:
            plot_res = False

        # Make a figure
        fig_height = 9.0 if plot_res else 5.0
        if has_alea:
            fig_height += 2
        fig, axs = plt.subplots(n_channels, 3, figsize=(13.5, fig_height), squeeze=False)
        gs = axs[0, 0].get_gridspec()
        for i in range(0, n_channels):
            axs[i, 0].remove()
        ax_map = fig.add_subplot(gs[:, 0])

        # Get the map and plot
        if "label" in test_out.keys():
            title_map = "True: "
            title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in test_out["label"][db_sample]])
        else:
            title_map = ""

        # Also: show prediction
        if not np.all(pred_fluxes == 0):
            pred_vals = pred_fluxes.mean(0)  # average over tau's - they shouldn't have any effect on FF prediction
            title_map += "\nPred: "
            title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in pred_vals])
            if has_alea:
                FF_STD = np.asarray([np.sqrt(np.diag(pred_fluxes_dict["covar"][_])) for _ in range(n_taus)]).mean(0)
                title_map += "\nSTD: "
                title_map += " ".join(["{:2.1f}%".format(100 * i).rjust(5, ' ') for i in FF_STD])

        if plot_res:
            plot_data_full = copy.copy(masked_to_full(np.mean(pred_fluxes_dict["count_maps"][:, :, 0], 0),
                                                      params["indexes"][0], nside=params["nside"]))
            # If model / res has multiple channels, corresponding to e.g. mean +- std FFs: average
            # if only 1 channel: has no effect anyway
            plot_data_model = copy.copy(masked_to_full(np.mean(np.mean(pred_fluxes_dict["count_maps_modelled_Poiss"], 0), 1),
                                                       params["indexes"][0], nside=params["nside"]))
            plot_data_res = copy.copy(masked_to_full(np.mean(np.mean(pred_fluxes_dict["count_maps_residual"], 0), 1),
                                                     params["indexes"][0], nside=params["nside"]))
            max_val = max(plot_data_model.max(), plot_data_full.max())
            if map_log:
                max_val = np.log10(1 + max_val)
        else:
            # Plot test_out["data"]: this is with exposure correction REMOVED if self.remove_exp, otherwise in terms of (normalised) counts
            plot_data_full = copy.copy(masked_to_full(test_out["data"][db_sample, :], params["indexes"][0], nside=params["nside"]))
            max_val = None

        axs_inds = [1, 4, 7] if plot_res else [1]
        for i_ind, ax_no in enumerate(axs_inds):
            max_val_loc = max_val
            min_val_loc = None
            cmap_loc = cmap_hp
            if i_ind == 0:
                plot_data_loc = plot_data_full
                title_loc = title_map
            elif i_ind == 1:
                plot_data_loc = plot_data_model
                title_loc = "Poissonian model (counts)"
            elif i_ind == 2:
                plot_data_loc = plot_data_res
                title_loc = "Model residual (counts)"
                max_val_loc = max_val / residual_clim_fac
                min_val_loc = - max_val / residual_clim_fac
                cmap_loc = cmap_hp_res
            plot_data_loc[total_mask_neg] = np.nan

            map_to_plot = np.log10(1 + plot_data_loc) if map_log else plot_data_loc
            hp.cartview(map_to_plot, nest=True, cmap=cmap_loc, badcolor="white", title=title_loc, cbar=False, fig=0,
                        sub=(3 if plot_res else 1, 3, ax_no), max=max_val_loc, min=min_val_loc)
            ax = plt.gca()
            ax.set_xlim([-params["outer_rad"] - 1, params["outer_rad"] + 1])
            ax.set_ylim([-params["outer_rad"] - 1, params["outer_rad"] + 1])
            if i_ind == 0 and has_alea:
                ax.set_title(ax.get_title(), fontsize=8)
            if i_ind == 0 and print_counts:
                ax.text(-params["outer_rad"], params["outer_rad"], "Sum:\n" + str(int(test_out["data"][db_sample, :].sum())),
                        va="top", ha="left")
                ax.text(params["outer_rad"], params["outer_rad"], "Max:\n" + str(int(test_out["data"][db_sample, :].max())),
                        va="top", ha="right")
        ax_map.axis("off")

        # Iterate over the channels
        for i_ch in range(n_channels):

            # Iterate over the taus
            for i_tau in range(n_taus):

                # Step plot for the differential histogram
                axs[i_ch, 2].step(bin_centres - width / 2.0, pred_hist[i_tau, :, i_ch], color=colors[i_tau], zorder=1, alpha=0.8, where="post")

                # For median: plot a triangle in addition
                if np.abs(tau_vec[i_tau] - 0.5) < 0.001:
                    axs[i_ch, 2].plot(bin_centres, pred_hist[i_tau, :, i_ch], markeredgecolor="k", color="white", marker="^", lw=0,
                                      zorder=5, alpha=0.8, markeredgewidth=1.5)

                # Plot cumulative histogram
                if i_tau < n_taus - 1:
                    # Draw the next section of the cumulative histogram in the right colour
                    if continuous_style:  # coloured area
                        axs[i_ch, 1].fill_between(x=bin_centres,
                                                  y1=pred_hist[i_tau, :, i_ch].cumsum(),
                                                  y2=pred_hist[i_tau + 1, :, i_ch].cumsum(), color=colors[i_tau], lw=0)
                    else:  # coloured bars
                        for i in range(len(bin_centres)):
                            # Draw the next section of the cumulative histogram in the right colour
                            axs[i_ch, 1].fill_between(x=[bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0],
                                                      y1=pred_hist[i_tau, :, i_ch].cumsum()[i],
                                                      y2=pred_hist[i_tau + 1, :, i_ch].cumsum()[i], color=colors[i_tau], lw=0)
                            # If highest ~0 or lowest ~1: plot a line to make the prediction visible
                            if i_tau == 0 and pred_hist[0, :, i_ch].cumsum()[i] > 0.99:
                                axs[i_ch, 1].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [1.0],
                                                  color=colors[0], lw=2, zorder=3)
                            elif i_tau == n_taus - 2 and pred_hist[-1, :, i_ch].cumsum()[i] < 0.01:
                                axs[i_ch, 1].plot([bin_centres[i] - width / 2.0, bin_centres[i] + width / 2.0], 2 * [0.0],
                                                  color=colors[-1], lw=2, zorder=3)

            if has_hist_label:
                # Bar plot for true differential histogram
                axs[i_ch, 2].bar(bin_centres, test_gce_hist[db_sample, :, i_ch], fc=cum_col_faint, ec=cum_col_1, width=width,
                                 lw=2, zorder=3)
                # Bar plot for true cumulative histogram
                axs[i_ch, 1].bar(bin_centres, test_gce_hist[db_sample, :, i_ch].cumsum(), fc=cum_col_faint, ec=cum_col_1, width=width, lw=2)

            if additional_histogram is not None:
                axs[i_ch, 2].bar(bin_centres, additional_histogram, fc="none", ec="silver", width=width, lw=2)
                axs[i_ch, 1].bar(bin_centres, additional_histogram.cumsum(), fc="none", ec="silver", width=width, lw=2)

            # Set axes limits
            axs[i_ch, 1].set_ylim([-0.075, 1.075])
            axs[i_ch, 2].set_ylim([-0.075, 1.075])

            axs[i_ch, 1].set_xticks(bin_centres[::2])
            for _ in [1, 2]:
                axs[i_ch, _].xaxis.set_major_locator(FixedLocator(bin_centres[0::2]))
                axs[i_ch, _].xaxis.set_minor_locator(FixedLocator(bin_centres[1::2]))
                axs[i_ch, _].xaxis.set_minor_formatter(FormatStrFormatter("%d"))
                axs[i_ch, _].tick_params(which='minor', axis='x')
                axs[i_ch, _].tick_params(which="minor", direction='out', length=4, pad=4)
                axs[i_ch, _].tick_params(which="major", direction='out', length=4)
                if params["which_histogram"] == 1:
                    x_tick_major_labs = np.round(bin_centres[0::2], 2)
                    x_tick_minor_labs = np.round(bin_centres[1::2], 2)
                    # Plot 1-ph line if mean exposure is given
                    if mean_exp > 0:
                        one_ph_flux = np.log10(1 / mean_exp)
                        axs[i_ch, _].axvline(one_ph_flux, color="orange", ls="--")
                        if _ == 1 and i_ch == 0:
                            axs[i_ch, _].text(x=one_ph_flux + 0.12, y=1.07, s="1 ph", rotation="horizontal", horizontalalignment='left',
                                              verticalalignment='top', fontsize=8, color="orange")

                        def F2S(x):
                            return 10 ** x * mean_exp
                        if i_ch == 0:
                            ax_exp_counts = axs[i_ch, _].twiny()
                            tick_locs = np.logspace(-2, 4, 7)
                            ax_exp_counts.set_xlim(F2S(np.asarray(axs[i_ch, _].get_xlim())))
                            ax_exp_counts.set_xticks(tick_locs)
                            ax_exp_counts.set_xticklabels(F2S(tick_locs))
                            ax_exp_counts.set_xlabel("Expected counts")
                            ax_exp_counts.set_xscale("log")
                else:
                    x_tick_major_labs = np.asarray([int(np.round(i, 0)) for i in 10 ** bin_centres[0::2]])
                    x_tick_minor_labs = np.asarray([int(np.round(i, 0)) for i in 10 ** bin_centres[1::2]])
                axs[i_ch, _].set_xticklabels(x_tick_major_labs, minor=False, rotation=90)
                axs[i_ch, _].set_xticklabels(x_tick_minor_labs, minor=True, rotation=90)
                # axs[_].set_xticklabels(np.asarray([int(np.round(i, 1 - int(np.ceil(np.log10(i))))) for i in 10 ** bin_centres[0::2]]), minor=False, rotation=90)
                # axs[_].set_xticklabels(np.asarray([int(np.round(i, 1 - int(np.ceil(np.log10(i))))) for i in 10 ** bin_centres[1::2]]), minor=True, rotation=90)

            # Template names
            axs[i_ch, 2].text(x=bin_centres[-1], y=1.0, s=temp_names[i_ch], rotation="horizontal", horizontalalignment='right',
                              verticalalignment='top', fontsize=12, color="k")

            all_titles = ["dN/dF", "Counts per PS", "Counts per pixel"]
            tit = all_titles[params["which_histogram"] - 1]
            if i_ch == n_channels - 1:
                if params["which_histogram"] == 1:
                    axs[i_ch, 1].set_xlabel(r"$\log_{10} \ F$")
                    axs[i_ch, 2].set_xlabel(r"$\log_{10} \ F$")
                else:
                    axs[i_ch, 1].set_xlabel("Count number")
                    axs[i_ch, 2].set_xlabel("Count number")
            # axs[i_ch, 1].set_ylabel("Cumulative histogram")
            # axs[i_ch, 2].set_ylabel("Histogram")
            if i_ch == 0:
                axs[i_ch, 1].set_title("Cumulative (" + tit + ")")
                axs[i_ch, 2].set_title("Differential (" + tit + ")")
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.0, wspace=0.2)

    # Save in a single PDF
    if len(name) > 0:
        filename = os.path.join(model.get_path("checkpoints"), name)
        multipage(filename)
        plt.close("all")


# Histograms
def EM1_distance(h1, h2, bins=None):
    """
    Calculate EM1 distance between two histograms.
    :param h1: histogram 1
    :param h2: histogram 2
    :param bins: histogram bins
    :return: EM1 (or 1-Wasserstein) distance
    """
    assert h1.shape == h2.shape, "Shape mismatch!"
    n_bins = h1.shape[0]
    if bins is None:
        bins = np.arange(n_bins)
    em1 = wasserstein_distance(u_values=bins, v_values=bins, u_weights=h1, v_weights=h2)
    all_in_bin_0 = np.zeros(n_bins)
    all_in_bin_0[0] = 1
    # em1_from_0_1 = stats.wasserstein_distance(u_values=bins, v_values=bins, u_weights=h1, v_weights=all_in_bin_0)
    # em1_from_0_2 = stats.wasserstein_distance(u_values=bins, v_values=bins, u_weights=h2, v_weights=all_in_bin_0)
    # print(em1_from_0_1, em1_from_0_2)
    # em1 = np.sum(np.abs(np.cumsum(h1) - np.cumsum(h2_cum))) in case of bins = np.arange(n_bins)
    return em1


def constrain_Poiss_count_fraction_from_counts_per_pixel(HIST, HIST_P, bin_centres):
    """
    Constrain the maximum Poissonian count fraction in a map by comparing the counts-per-pixel histogram (without PSF!)
    to its counterpart for a purely Poissonian map.
    :param HIST: counts-per-pixel histogram (might be Poiss. and/or Non-Poiss.): NOT normalised!
    :param HIST_P: counts-per-pixel histogram for a purely Poiss. map (should have same expected number of counts): NOT normalised!
    :param bin_centres: bin centres
    :return: Poiss. count ratio that minimised EM1 distance to Poisson. histogram, corresponding EM1 distance
    """
    # Calculate the flux fraction consistent with Poissonian emission
    counts_in_map = HIST.sum()

    HIST_NEW = HIST.copy()
    hist_p = HIST_P / HIST_P.sum()

    counter = 0
    done = False

    count_ratio_removed_to_all_best = 1.0
    em1_P_NP_brightest_removed_best = np.infty

    for i_bin in range(len(bin_centres))[::-1]:
        while HIST_NEW[i_bin].sum() > 0:
            count_ratio_removed_to_all = HIST_NEW.sum() / HIST.sum()
            hist_np_new = HIST_NEW / HIST_NEW.sum()
            em1_P_NP_brightest_removed = EM1_distance(hist_p, hist_np_new, bin_centres)
            if em1_P_NP_brightest_removed < em1_P_NP_brightest_removed_best:
                em1_P_NP_brightest_removed_best = em1_P_NP_brightest_removed
                count_ratio_removed_to_all_best = count_ratio_removed_to_all
            counter += 1
            # remove 1 count in highest bin
            HIST_NEW[i_bin] -= 1
            if HIST_NEW.sum() == 0 or counter >= counts_in_map:
                done = True
                break
        if done:
            break

    return count_ratio_removed_to_all_best, em1_P_NP_brightest_removed_best


# # # # # # # # # # # # # # # # # #
# Likelihood-to-evidence ratio functions
# # # # # # # # # # # # # # # # # #
# PDF Sampler
class PDFSampler:
    def __init__(self, xvals, pofx):
        """ At outset sort and calculate CDF so not redone at each call

            :param xvals: array of x values
            :param pofx: array of associated p(x) values (does not need to be
                   normalised)
        """
        self.xvals = xvals
        self.pofx = pofx

        # Check p(x) >= 0 for all x, otherwise stop
        assert(np.all(pofx >= 0)), "pdf cannot be negative"

        # Sort values by their p(x) value, for more accurate sampling
        self.sortxvals = np.argsort(self.pofx)
        self.pofx = self.pofx[self.sortxvals]

        # Calculate cdf
        self.cdf = np.cumsum(self.pofx)

    def __call__(self, samples):
        """ When class called returns samples number of draws from pdf

            :param samples: number of draws you want from the pdf
            :returns: number of random draws from the provided PDF

        """

        # Random draw from a uniform, up to max of the cdf, which need
        # not be 1 as the pdf does not have to be normalised
        unidraw = np.random.uniform(high=self.cdf[-1], size=samples)
        cdfdraw = np.searchsorted(self.cdf, unidraw)
        cdfdraw = self.sortxvals[cdfdraw]
        return self.xvals[cdfdraw]


def get_A_from_z(z, prior_lims, A_is_log=False):
    """
    Function to get value from z and the prior range for Poissonian templates
    :param z: array of shape n_maps x n_models, with elements between 0 and 1
    :param prior_lims: prior limits: [lower_lim, upper_lim]
    :param A_is_log: if True: prior limits refer to log10(A) instead of A
    :return: model intensity A (same shape as z)
    """
    lower_prior_lim, upper_prior_lim = prior_lims[:, 0], prior_lims[:, 1]
    if A_is_log:
        A = (1 - z) * (10 ** lower_prior_lim) + z * (10 ** upper_prior_lim)
    else:
        A = (1 - z) * lower_prior_lim + z * upper_prior_lim

    return A


def plot_posterior(model, true_map, n_params, n_points_posterior=256, nside=128, truth=None, filename=None, sess=None,
                   titles=None, pp=None, exp=None, template_maps=None, n_samples_FF=2048, colours_FF=None, filename_FF=None,
                   n_bins_FF=51, FF_range=None, plot_z=False):

    if exp is not None and template_maps is not None and pp is not None:
        compute_FF_posteriors = True
        if np.any([k != "A" for k in pp.keys]):
            compute_FF_posteriors = False
            # TODO: SO FAR, ONLY POISSONIAN!
            print("FF posteriors are not yet implemented for Non-Poissonian templates!!!")
    else:
        compute_FF_posteriors = False

    if colours_FF is None:
        colours_FF = ['#ff0000', '#ec7014', '#fec400', '#37c837', 'deepskyblue', 'darkslateblue', 'k']

    if len(true_map.shape) == 1:
        true_map = true_map[None]
    tiled_map = np.tile(true_map, [n_points_posterior, 1])
    rand_z_posterior = np.random.uniform(0, 1, [n_points_posterior, n_params])
    test_map_dict = {"data": tiled_map, "label": rand_z_posterior}
    pred_test_map_dict = model.predict(test_map_dict, sess=sess, only_last_layer=False, tau_hist=None)

    n_col = max(int(np.ceil(np.sqrt(n_params))), 1)
    n_row = int(np.ceil(n_params / n_col))

    if filename is not None:
        old_interactive_state = plt.isinteractive()
        plt.ioff()
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4))
    axs_f = axs.flatten()

    if compute_FF_posteriors:
        intensity_arr = np.zeros((n_samples_FF, n_params))
        if truth is not None:
            intensity_truth = np.zeros((n_params))

    for i_param in range(n_col * n_row):
        if i_param >= n_params:
            axs_f[i_param].axis("off")
            continue
        this_z = rand_z_posterior[:, i_param].copy()
        sort_inds = np.argsort(this_z)
        this_z, this_logR = this_z[sort_inds], pred_test_map_dict["logits_mean"][sort_inds, i_param]
        this_R = np.exp(this_logR)
        I = trapz(this_R, this_z)
        p = this_R / I  # normalise such that integral(p) = 1
        property_to_plot = this_z if pp is None or plot_z else pp.unnormalise(i_param, this_z)
        axs_f[i_param].plot(property_to_plot, p)
        if truth is not None:
            truth_to_plot = truth[i_param] if pp is None or plot_z else pp.unnormalise(i_param, truth[i_param])
            axs_f[i_param].axvline(truth_to_plot, color="orange", ls="--")
        if titles is not None:
            axs_f[i_param].set_title(titles[i_param])
        if pp is None or plot_z:
            axs_f[i_param].set_xlabel("z")
        else:
            axs_f[i_param].set_xlabel(pp.parameter_names[i_param])
        axs_f[i_param].set_ylabel("Prob.")

        # FF posteriors? TODO: exp. corr. removed here or not?
        if compute_FF_posteriors:
            this_PDF_sampler = PDFSampler(this_z, this_R)
            these_samples_z = this_PDF_sampler(n_samples_FF)
            these_samples_A = pp.unnormalise(i_param, these_samples_z)
            intensity_arr[:, i_param] = return_intensity_arrays_poiss(these_samples_A, template_maps[i_param, :], exp,
                                                                      counts=False, nside=nside, a_is_log=False)
            if truth is not None:
                true_A = pp.unnormalise(i_param, truth[i_param])
                intensity_truth[i_param] = return_intensity_arrays_poiss(np.asarray([true_A]), template_maps[i_param, :], exp,
                                                                         counts=False, nside=nside, a_is_log=False)

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    # FF posterior plot
    if compute_FF_posteriors:
        FFs = intensity_arr / intensity_arr.sum(1, keepdims=True)
        if truth is not None:
            FFs_truth = intensity_truth / intensity_truth.sum(0, keepdims=True)
        qs = [0.16, 0.5, 0.84]
        FF_bins = np.linspace(0, 100, n_bins_FF)
        frac_hist_comp = np.asarray([np.histogram(100 * np.array(FFs[:, i_param]), bins=FF_bins, range=(0, 100), density=True)[0] \
                                     for i_param in range(n_params)])
        FF_bin_centres = (FF_bins[1:] + FF_bins[:-1]) / 2.0

        fig_FF, ax_FF = plt.subplots(1, 1, figsize=(14, 10))
        y_min, y_max = -0.02, frac_hist_comp.max() + 0.02
        for i_param in range(n_params):
            ax_FF.plot(FF_bin_centres, frac_hist_comp[i_param, :], color=colours_FF[i_param], lw=2, label="FF"+str(i_param))
            these_quantiles = 100 * np.quantile(FFs[:, i_param], qs)
            for quant in these_quantiles:
                ax_FF.plot([quant] * 2, [0, y_max - 0.02], ls="--", color=colours_FF[i_param])
            if truth is not None:
                ax_FF.plot(100 * FFs_truth[i_param], y_max, ls="none", marker="v", color=colours_FF[i_param], markersize=10,
                           mec="k")
                ax_FF.plot(100 * FFs_truth[i_param], y_min, ls="none", marker="^", color=colours_FF[i_param], markersize=10,
                           mec="k")
        if titles is not None:
            FF_handles, _ = ax_FF.get_legend_handles_labels()
            ax_FF.legend(FF_handles, titles)
        ax_FF.set_xlabel("Flux fraction [%]")
        ax_FF.set_ylabel("Prob.")

        if FF_range is None:
            histogram_sum = np.mean(frac_hist_comp.sum(1))
            eps = 1e-4
            FF_range = [0, FF_bin_centres[np.apply_along_axis(lambda x: np.argmax(x.cumsum() > histogram_sum - eps), 1, frac_hist_comp) + 1].max()]
            ax_FF.set_xlim(FF_range)
        else:
            ax_FF.set_xlim(FF_range)

        ax_FF.plot(FF_range, [0, 0], color="k", lw=2.1)
        ax_FF.set_ylim(-0.03, y_max + 0.01)

        if filename_FF is not None:
            fig_FF.savefig(filename_FF, bbox_inches="tight")
            plt.close(fig_FF)

    if old_interactive_state:
        plt.ion()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # U-Net specific part # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def unet_evaluate(true, pred, models, model_names, UM, zoom_xlim=None, zoom_ylim=None, marg=None, figsize=(16, 10),
                  cmap="rocket", plot_inds=[0], folder=None):
    """
    Analysis function for the U-Net
    :param true: true counts (format: n_batch x n_pix x n_templates)
    :param pred: predicted counts (format: n_batch x n_pix x n_templates)
    :param models: templates (array or list)
    :param model_names: template names (array or list)
    :param UM: lambda function that unmasks the maps to the entire sky
    :param zoom_xlim: ax. zoom region for x-coord.
    :param zoom_ylim: ax. zoom region for y-coord.
    :param marg: margin for the axes
    :param figsize: size of the figure
    :param cmap: colour map
    :param plot_inds: list of indices to plot (w.r.t. to batch size dimension)
    :param folder: folder for saving plots
    """
    map_tot = true.sum(2, keepdims=True)
    n_templates = true.shape[2]

    # 1) Calculate total count fractions, print stats, and show plot
    pred_CFs = np.asarray([pred[:, :, i].sum(1) / pred.sum(-1).sum(-1) for i in range(pred.shape[2])]).T
    true_CFs = np.asarray([true[:, :, i].sum(1) / true.sum(-1).sum(-1) for i in range(true.shape[2])]).T
    print("Count fraction statistics for the entire map:")
    print_stats(pred_CFs, true_CFs, models)
    out_file = None if folder is None else os.path.join(folder, "count_fraction_error.pdf")
    make_error_plot(models, true_CFs, pred_CFs, delta_y=-0.13, model_names=model_names, out_file=out_file)

    # 2) Calculate pixelwise stats
    eps = 1e-12
    relevant_pixels = (map_tot > 0).astype(np.float32)  # pixels with at least 1 count
    scale_fac_l1 = 1 / (np.sqrt(map_tot + eps))
    scale_fac_l2 = 1 / (map_tot + eps)
    l2_val_loss = np.mean((true - pred) ** 2 * scale_fac_l2 * relevant_pixels, 1)
    l1_val_loss = np.mean(np.abs(true - pred) * scale_fac_l1 * relevant_pixels, 1)
    print(model_names)
    print("Mean l1 loss:", l1_val_loss.mean(0))
    print("Mean l2 loss:", l2_val_loss.mean(0))

    # 3) Plot some examples
    if zoom_xlim is None:
        zoom_xlim = [-0.3, 0.3]
    if zoom_ylim is None:
        zoom_ylim = [-0.4, 0.4]
    if marg is None:
        marg = 0.02

    def zoom_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.axis("off")
        mw_ax = plt.gca()
        if zoom_xlim is not None:
            mw_ax.set_xlim(zoom_xlim)
        if zoom_ylim is not None:
            mw_ax.set_ylim(zoom_ylim)

    for ind in plot_inds:
        fig_ex, axs_ex = plt.subplots(3, n_templates + 1, figsize=figsize)
        hp.mollview(UM(map_tot[ind, :, 0]),
                    nest=True, sub=[3, n_templates + 1, 1], fig=fig_ex,
                    title="Map " + str(ind), margins=4 * [marg], cbar=True, cmap=cmap)
        zoom_axis(axs_ex[0, 0])
        hp.mollview(UM(relevant_pixels[ind, :, 0]),
                    nest=True, sub=[3, n_templates + 1, n_templates + 2], fig=fig_ex,
                    title="Pixels with counts", margins=4 * [marg], cbar=True, cmap=cmap)
        zoom_axis(axs_ex[1, 0])
        zoom_axis(axs_ex[2, 0])

        for i_temp, temp in enumerate(models):
            max_c = int(np.ceil(max(np.abs(pred[ind, :, i_temp]).max(), np.abs(true[ind, :, i_temp]).max())))
            ax_ind_true = np.ravel_multi_index([0, i_temp], dims=[3, n_templates + 1], order="C") + 2
            ax_ind_pred = np.ravel_multi_index([1, i_temp], dims=[3, n_templates + 1], order="C") + 2
            ax_ind_diff = np.ravel_multi_index([2, i_temp], dims=[3, n_templates + 1], order="C") + 2
            hp.mollview(UM(true[ind, :, i_temp]),
                        nest=True, sub=[3, n_templates + 1, ax_ind_true], fig=fig_ex,
                        title="%3.3f" % true_CFs[ind, i_temp], margins=4 * [marg], cbar=True, cmap=cmap, max=max_c)
            zoom_axis(axs_ex[0, i_temp + 1])

            hp.mollview(UM(pred[ind, :, i_temp]),
                        nest=True, sub=[3, n_templates + 1, ax_ind_pred], fig=fig_ex,
                        title="%3.3f" % pred_CFs[ind, i_temp], margins=4 * [marg], cbar=True, cmap=cmap, max=max_c)
            zoom_axis(axs_ex[1, i_temp + 1])
            max_delta = int(np.ceil(np.abs(pred[ind, :, i_temp] - true[ind, :, i_temp]).max()))
            hp.mollview(UM(pred[ind, :, i_temp] - true[ind, :, i_temp]),
                        nest=True, sub=[3, n_templates + 1, ax_ind_diff], fig=fig_ex,
                        title=model_names[i_temp], margins=4 * [marg], cbar=True, cmap="seismic", max=max_delta, min=-max_delta)
            zoom_axis(axs_ex[2, i_temp + 1])

        pretty_plots()
        plt.tight_layout()

        if folder is not None and len(plot_inds) > 0:
            fig_ex.savefig(os.path.join(folder, "example_prediction" + str(ind) + ".pdf"))

    # 4) Plot how many Poisson sigmas the estimates in each pixel are off
    cmap_err = plt.cm.get_cmap('seismic', 7)
    for ind in plot_inds:
        fig_err, axs_err = plt.subplots(1, n_templates, figsize=figsize)

        for i_temp, temp in enumerate(models):
            delta_in_sigmas = (pred[ind, :, i_temp] - true[ind, :, i_temp]) / (np.sqrt(true[ind, :, i_temp]) + 1e-15)
            hp.mollview(UM(delta_in_sigmas),
                        nest=True, sub=[1, n_templates, i_temp + 1], fig=fig_err,
                        title=model_names[i_temp], margins=4 * [marg], cbar=True, cmap=cmap_err, min=-3, max=3)
            zoom_axis(axs_err[i_temp])

        pretty_plots()
        plt.tight_layout()

        if folder is not None and len(plot_inds) > 0:
            fig_err.savefig(os.path.join(folder, "poisson_error_" + str(ind) + ".pdf"))


    # 5) Print pixel-wise stats. in terms of Poisson sigmas
    print("Average number of pixel estimates within 1, 2, 3 true Poisson sigmas:")
    for i_temp, temp in enumerate(models):
        delta_in_sigmas = (pred[:, :, i_temp] - true[:, :, i_temp]) / (np.sqrt(true[:, :, i_temp]) + 1e-15)
        print("  ", model_names[i_temp] + ":")
        for i_sigma in [1, 2, 3]:
            print("    " + str(i_sigma) + " sigma:", "%1.3f" % ((np.abs(delta_in_sigmas) <= i_sigma).sum() / (1 - np.isnan(delta_in_sigmas)).sum()))


# [["dif_O_pibs", "dif_A_pibs"], ["dif_O_ic", "dif_A_ic"], ["bub"], ...] -> ["dif_O_pibs", "dif_A_pibs", "dif_O_ic", "dif_A_ic", "bub", ...]
def flatten_var_fracs(fracs_stacked):
    return np.hstack([*fracs_stacked])


# ["dif_O_pibs", "dif_A_pibs", "dif_O_ic", "dif_A_ic", "bub", ...] -> [["dif_O_pibs", "dif_A_pibs"], ["dif_O_ic", "dif_A_ic"], ["bub"], ...]
def stack_var_fracs(fracs_flat, model_vars):
    fracs_stacked = [None] * len(model_vars)
    i_count = 0
    for i_temp in range(len(model_vars)):
        fracs_stacked[i_temp] = fracs_flat[i_count:i_count+len(model_vars[i_temp])]
        i_count += len(model_vars[i_temp])
    return fracs_stacked


# Extract the template variant fractions for which more then one variant is present
def get_relevant_var_fracs(var_fracs, model_vars, models):
    var_fracs_rel = []
    models_rel = []
    indices_rel = []
    i_count = 0
    for i_temp, temp in enumerate(model_vars):
        if len(model_vars[i_temp]) > 1:
            var_fracs_rel.append(var_fracs[:, i_count:i_count+len(model_vars[i_temp])])
            i_count += len(model_vars[i_temp])
            models_temp = []
            for j in range(1, len(model_vars[i_temp])+1):
                models_temp.append(models[i_temp] + " " + str(j))
            models_rel.append(models_temp)
            indices_rel.append(i_temp)
        else:
            i_count += len(model_vars[i_temp])
    return var_fracs_rel, models_rel, indices_rel
