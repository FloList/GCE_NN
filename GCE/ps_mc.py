# Script that does point source Monte Carlo based off of user defined source
# count distribution, template, exposure map, and user defined point source
# function. Can save result of simulation to .npy file.
# Modified from https://github.com/nickrodd/NPTFit-Sim/tree/master/NPTFit-Sim.
###############################################################################
import sys
import time
import numpy as np
import healpy as hp
import scipy.stats as stats
# import warnings


# Random upscaling:
# PS positions are determined in 2 steps:
#   1. Draw from a multinomial distribution where the probabilities are defined by the template to determine the pixel
#   2. Shift each PS to the centre of a randomly determined high-res. pixel contained within the original low-res. pixel
# The following function takes care of step 2.
def random_u_grade_ang(m_inds, nside_in=0, nside_out=16384, is_nest=False):
    """
    Random upscaling of the PS positions, given the pixel centres at resolution nside_in.
    Each PS is moved to one of the high-resolution pixel centres at resolution nside_out.
    For example, for nside_in = 128 and nside_out = 16384, there are npix_out / npix_in = 16384
    possible PS locations within each pixel.
    :param m_inds: indices of the PSs w.r.t. nside_in, in RING ordering
    :param nside_in: nside in
    :param nside_out: nside to use for upscaling
    :param is_nest: if True: indices are assumed to correspond to NEST format instead of RING
    :return: theta, phi of randomly placed PSs within each pixel
    """
    if len(m_inds) == 0:
        return m_inds
    n_ps = len(m_inds)  # number of point sources
    if is_nest:
        m_inds_nest = m_inds
    else:
        m_inds_nest = hp.ring2nest(nside_in, m_inds)  # convert to NEST ordering
    hp.isnsideok(nside_out, nest=True)  # check that nside_out is okay
    npix_in = hp.nside2npix(nside_in)
    npix_out = hp.nside2npix(nside_out)
    rat2 = npix_out // npix_in
    # For each PS, draw a random fine pixel within the coarse pixel
    inds_fine = np.random.choice(rat2, size=n_ps)
    # Set indices w.r.t. upscaled nside (NEST): rat2 * m_inds_nest -> high-res. pixel 0, inds_fine adds 0 ... rat2 - 1
    inds_out = rat2 * m_inds_nest + inds_fine
    # Calculate angles
    th_out, ph_out = hp.pix2ang(nside_out, inds_out, nest=True, lonlat=False)
    # Note: for rat2 = 16384 and 150 PSs, the chance that there exists at least a pair of equal th/ph_out is > 50%!
    return th_out, ph_out


def run(flux_arr, temp, exp, weights, pdf_psf_sampler, name="map", save=False, getnopsf=False, getcts=False,
        upscale_nside=16384, verbose=False, clean_count_list=False, inds_outside_roi=None, is_nest=False):
    """
    Runs point source Monte Carlo by reading in template, source count distribution parameters, exposure
    map, and the user defined PSF.
    :param flux_arr: array of fluxes of sources to generate
    :param temp: HEALPix numpy array of template
    :param exp: HEALPix numpy array of exposure map
    :param weights: array of weights for each spectral bin, i.e. normalised spectrum
    :param pdf_psf_sampler: user-defined PSF: object of type PDFSampler (see class above). Can be None: no PSF.
    :param name: string for the name of output .npy file
    :param save: option to save map to .npy file
    :param getnopsf: return the map that would result without the PSF (as a second channel)
    :param getcts: return the array of counts drawn
    :param upscale_nside: nside to use for randomly determining the PS location *within* each pixel
    :param verbose: print when starting and finishing
    :param clean_count_list:
        if True: photons that are smeared to pixels with temp == 0 by the PSF are removed from the count list,
                 assuming that a mask will be applied later that masks pixels where temp == 0
    :param inds_outside_roi:
        if not None: set of indices where PSs can be located (and smear counts into the ROI); however, these pixels lie
                     outside ROI and will be masked later. PSs located at these indices will be deleted from the
                     returned flux array and photon count list.
                     NOTE: This requires clean_count_list = False!
    :param is_nest: if True: temp and exp are assumed to be in NEST format instead of RING (output: same format!)
    :returns: HEALPix format numpy array of simulated map (if getnopsf == True: shape: npix x 2, else: npix),
              list of counts for each PS (if desired), and flux array containing flux in ROI if inds_outside_roi != None
    """

    # Generate simulated counts map
    map_arr, map_arr_no_psf, cts_arr, flux_arr_out = make_map(flux_arr, temp, exp, weights, pdf_psf_sampler,
                                                              upscale_nside, verbose, clean_count_list, inds_outside_roi,
                                                              is_nest)

    # Save the file as an .npy file
    if save:
        np.save(str(name) + ".npy", np.array(map_arr).astype(np.int32))

    if verbose:
        print("Done simulation.")

    if getnopsf:
        map_arr_return = np.concatenate([np.expand_dims(np.array(map_arr).astype(np.int32), -1),
                                         np.expand_dims(np.array(map_arr_no_psf).astype(np.int32), -1)], axis=-1)
    else:
        map_arr_return = np.array(map_arr).astype(np.int32)

    if getcts:
        if inds_outside_roi is not None:
            return map_arr_return, np.array(cts_arr).astype(np.int32), flux_arr_out
        else:
            return map_arr_return, np.array(cts_arr).astype(np.int32)
    else:
        if inds_outside_roi is not None:
            return map_arr_return, flux_arr_out
        else:
            return map_arr_return


def make_map(flux_arr, temp, exp, weights, pdf_psf_sampler, upscale_nside=16384, verbose=False, clean_count_list=True,
             inds_outside_roi=None, is_nest=False):
    """
    Given an array of fluxes for each source, template & exposure map, and user defined PSF, simulates and returns
      1) a counts map,
      2) the counts map before the PSF was applied,
      3) the list of the photons per PS (len: # PSs, sum: # counts in the map)
    The source positions are determined by
      1) drawing from a multinomial distribution as determined by the template
      2) randomly determining the PS position within each pixel for each PS
    :param flux_arr: list/array for source fluxes
    :param temp: array of template: MUST BE NORMALISED TO SUM UP TO UNITY!
    :param exp: array of exposure map
    :param weights: array of weights for each spectral bin, i.e. normalised spectrum
    :param pdf_psf_sampler: user-defined PSF: object of type PDFSampler (see class above). Can be None (no PSF).
    :param upscale_nside: nside to use for randomly determining the PS location *within* each pixel
    :param verbose: print when starting
    :param clean_count_list:
        if True: photons that are smeared to pixels with temp == 0 by the PSF are removed from the count list,
                 assuming that a mask will be applied later that masks pixels where temp == 0
    :param inds_outside_roi:
        if not None: set of indices where PSs can be located (and smear counts into the ROI); however, these pixels lie
                     outside ROI and will be masked later. PSs located at these indices will be deleted from the
                     returned flux array and photon count list.
                     NOTE: This requires clean_count_list = False!
    :param is_nest: if True: temp and exp are assumed to be in NEST format instead of RING (output: same format!)
    :returns: array of simulated counts map, the same map before applying the PSF, list of counts for each PS,
              flux array (which is identical to input, except when inds_outside_roi is provided)

    """
    if inds_outside_roi is not None:
        assert not clean_count_list, "'clean_count_list' is not supported if 'inds_outside_roi' is provided!"
    assert not clean_count_list, "clean_count_list with energy bins is not implemented!"

    # exp.setflags(write=1)  # these numpy arrays may be read-only! need to change this for ray!
    # temp.setflags(write=1)
    if type(flux_arr) == list:
        flux_arr = np.asarray(flux_arr)
    # flux_arr.setflags(write=1)

    n = len(flux_arr)  # number of PSs
    npix, nbins = temp.shape
    nside = hp.npix2nside(npix)
    flux_arr_return = np.asarray([])
    flux_arr_in_roi = np.asarray([])
    num_phot_in_roi = np.asarray([])

    # Initialise the map
    map_arr = np.zeros((npix, nbins), dtype=np.int32)
    map_arr_no_psf = np.zeros((npix, nbins), dtype=np.int32)

    if verbose:
        print("Simulating counts maps ...")

    # Check that template is normalised such that it sums up to unity.
    # Do NOT allow to normalise within this function because of parallelisation (could change template in memory)
    assert(np.abs(np.sum(temp) - 1) < 1e-5), "Template is not normalised!"

    # Draw pixel positions of the PSs and get an array with the (possibly multiple) indices
    inds_ps_bool = stats.multinomial.rvs(p=temp.sum(1), n=n)  # boolean array: inds_PS_bool.sum() == n.  NOTE: summing over energies here; weights will take care of the energy distribution
    inds_ps = np.repeat(np.arange(npix), inds_ps_bool)  # array with indices: len(inds_PS) == n

    # If no PSs: return at this point
    if len(inds_ps) == 0:
        return map_arr, map_arr_no_psf, [], []

    # Randomly shift the PS centres by upscaling to higher nside and randomly moving the centres to those at higher res:
    # this means that for e.g. nside = 128 and upscale_nside = 16384, there are upscale_npix / npix = 16384 possible
    # locations for each PS within each nside = 128 pixel.
    # Note that the HEALPix tesselation is equal area -> (discretised) uniform PDF within each pixel.

    th, ph = random_u_grade_ang(inds_ps, nside_in=nside, nside_out=upscale_nside, is_nest=is_nest)

    # Determine the pixels corresponding to (th, ph):
    # should not have changed since we're only moving the PS within each pixel!
    pix_ps = hp.ang2pix(nside, th, ph, nest=is_nest)
    assert np.all(pix_ps == inds_ps), "The PSs are not in the correct pixels anymore!"

    # Find expected number of source photons and then do a Poisson draw.
    # Weight the total flux by the expected flux in that bin
    num_phot = np.random.poisson(flux_arr[:, None] * weights * exp[pix_ps])

    # if actual ROI is subset: set flux array of the PSs in the ROI, as well as num_phot
    # if use_numba:
    #     @numba.njit
    #     def get_inds_ps_not_in_roi(inds_ps, inds_outside_roi):
    #         out = []
    #         for counter, ind in enumerate(inds_ps):
    #             if ind in inds_outside_roi:
    #                 out.append(counter)
    #         return np.asarray(out)
    #
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")  # using sets in Numba is deprecated and will eventually be replaced by numba.typed.Set
    #         # https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    #         inds_ps_not_in_roi = get_inds_ps_not_in_roi(inds_ps, inds_outside_roi)
    #
    # else:
    # Fallback if Numba is not available
    # NOTE: inds_outside_roi should be a SET, gives great speed-up for member search!
    inds_ps_not_in_roi = np.asarray([counter for (counter, ind) in enumerate(inds_ps) if ind in inds_outside_roi])

    if inds_outside_roi is not None:
        if len(inds_ps_not_in_roi) > 0:
            flux_arr_in_roi = np.delete(flux_arr, inds_ps_not_in_roi, axis=0)
            num_phot_in_roi = np.delete(num_phot, inds_ps_not_in_roi, axis=0)
        else:
            flux_arr_in_roi = flux_arr
            num_phot_in_roi = num_phot

    # Get array containing the pixel of each count before applying the PSF
    pix_counts = [np.repeat(pix_ps, num_phot[:, i]) for i in range(nbins)]

    # if no PSF:
    if pdf_psf_sampler is None:
        # posit = pix_counts
        # num_phot_cleaned = num_phot
        raise NotImplementedError

    # PSF correction:
    else:
        # Create a rotation matrix for each source.
        # Shift phi coord pi/2 to correspond to center of HEALPix map during rotation.
        phm = ph + np.pi / 2.

        # Each source is initially treated as being located at theta=0, phi=0 as
        # the drawn PSF distances simply correspond to photon theta position.
        # A random phi value [0, 2pi] is then drawn. Each photon is then rotated
        # about the x axis an angle corresponding to the true theta position of
        # the source, followed by a rotation about the z axis by the true phi
        # position plus an additional pi/2 radians.
        a0 = np.zeros(n)
        a1 = np.ones(n)
        rotx = np.array([[a1, a0, a0], [a0, np.cos(th), -np.sin(th)], [a0, np.sin(th), np.cos(th)]])
        rotz = np.array([[np.cos(phm), -np.sin(phm), a0], [np.sin(phm), np.cos(phm), a0], [a0, a0, a1]])

        # Sample distances from PSF for each source photon.
        n_counts_tot = num_phot.sum(0)  # total counts per bin
        dist_flat = [sampler(n) for sampler, n in zip(pdf_psf_sampler, n_counts_tot)]  # list of distances for flattened counts, len: n_counts_tot
        assert all(len(d) == n_counts_tot[i] for i, d in enumerate(dist_flat)), \
            "Length of distance list does not match total counts!"

        for i_bin in range(nbins):
            # Reshape: 3 x 3 x N  ->  N x 3 x 3, then tile: one matrix for each count -> num_phot x 3 x 3
            rotx_tiled = np.repeat(np.transpose(rotx, [2, 0, 1]), num_phot[:, i_bin], axis=0)
            rotz_tiled = np.repeat(np.transpose(rotz, [2, 0, 1]), num_phot[:, i_bin], axis=0)

            # Draw random phi positions for all the counts from [0, 2pi]
            rand_phi = 2 * np.pi * np.random.uniform(0.0, 1.0, n_counts_tot[i_bin])

            # Convert the theta and phi to x,y,z coords.
            x = np.array(hp.ang2vec(dist_flat[i_bin], rand_phi))

            # Rotate coords over the x axis (first: make X a matrix for each count)
            xp = rotx_tiled @ np.expand_dims(x, -1)

            # Rotate again, over the z axis (then: remove the additional dimension to get a 3d-vector for each count)
            xp = np.squeeze(rotz_tiled @ xp, -1)

            # Determine pixel location from x,y,z values.
            posit = hp.vec2pix(nside, xp[:, 0], xp[:, 1], xp[:, 2], nest=is_nest)

            # if desired: clean num_phot by removing the counts that leaked into pixels where temp == 0
            # if clean_count_list:
            #     # Initialise the cleaned list
            #     num_phot_cleaned = num_phot.copy()
            #     # Get the global indices of counts that have leaked out
            #     counts_leaked_outside_roi = np.argwhere(temp[posit] == 0).flatten()
            #     if len(counts_leaked_outside_roi) > 0:
            #         # Reverse the "repeat" (except where num_count == 0, but these PSs are irrelevant)
            #         reverse_repeat = np.cumsum(num_phot) - 1
            #         # Find PSs from which counts have leaked outside of the ROI (PSs may be contained multiple times)
            #         leaked_ps = np.asarray([np.argmax(ind_rep <= reverse_repeat)
            #                                  for ind_rep in counts_leaked_outside_roi]).flatten()
            #         # Check that the PS indices refer to the same pixels as the global count indices
            #         assert np.all(pix_counts[counts_leaked_outside_roi] == pix_ps[leaked_ps]), \
            #             "Count removal went wrong! Aborting!"
            #         # Remove the leaked counts
            #         np.add.at(num_phot_cleaned, leaked_ps, int(-1))
            #         assert np.min(num_phot_cleaned) >= 0, "Negative counts encountered! Aborting!"
            #     # Return input flux array
            #     flux_arr_return = flux_arr
            # else:

            if inds_outside_roi is not None:
                num_phot_cleaned = num_phot_in_roi
                flux_arr_return = flux_arr_in_roi
            else:
                num_phot_cleaned = num_phot
                flux_arr_return = flux_arr

            # Add all the counts: note: use "at" such that multiple counts in a pixel are added
            # if use_numba:
            #     @numba.njit
            #     def add_fun(m, p):
            #         for pp in p:
            #             m[pp] += 1
            #         return m
            #
            #     map_arr = add_fun(map_arr, posit)
            #     map_arr_no_psf = add_fun(map_arr_no_psf, pix_counts)
            #
            # else:
            # Fallback without Numba
            np.add.at(map_arr[:, i_bin], posit, int(1))  # pixels AFTER PSF
            np.add.at(map_arr_no_psf[:, i_bin], pix_counts[i_bin], int(1))  # pixels BEFORE PSF

    # Return map, map before PSF, and num_phot_cleaned
    return map_arr, map_arr_no_psf, num_phot_cleaned, flux_arr_return
