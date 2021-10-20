###############################################################################
#
# Difference w.r.t. ps_mc_fast.py:
# Samples PS on the entire sky and afterwards extracts the selected region.
# Like this, emission leaks out of and into the ROI.
#
###############################################################################
import numpy as np
import healpy as hp
import scipy.stats as stats
# import ray


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


# Random upscaling:
# PS positions are determined in 2 steps:
#   1. Draw from a multinomial distribution where the probabilities are defined by the template to determine the pixel
#   2. Shift each PS to the centre of a randomly determined high-res. pixel contained within the original low-res. pixel
# The following function takes care of step 2.
def random_u_grade_ang(m_inds, nside_in=0, nside_out=16384, is_nest=False):
    """
    Random upscaling of the PS positions, given the pixel centres at resolution nside_in.
    Each PS is moved to one of the high-resolution pixel centres at resolution nside_out.
    For example, for nside_in = 128 and nside_out = 16384, there are npix_out / npix_in = 16384 possible PS locations
    within each pixel.
    :param m_inds: indices of the PSs w.r.t. nside_in, in RING ordering
    :param nside_in: nside in
    :param nside_out: nside to use for upscaling
    :param is_nest: if True: indices are assumed to correspond to NEST format instead of RING
    :return: theta, phi of randomly placed PSs within each pixel
    """
    if len(m_inds) == 0:
        return m_inds
    n_PS = len(m_inds)  # number of point sources
    if is_nest:
        m_inds_nest = m_inds
    else:
        m_inds_nest = hp.ring2nest(nside_in, m_inds)  # convert to NEST ordering
    hp.isnsideok(nside_out, nest=True)  # check that nside_out is okay
    npix_in = hp.nside2npix(nside_in)
    npix_out = hp.nside2npix(nside_out)
    rat2 = npix_out // npix_in
    # For each PS, draw a random fine pixel within the coarse pixel
    inds_fine = np.random.choice(rat2, size=n_PS)
    # Set indices w.r.t. upscaled nside (NEST): rat2 * m_inds_nest -> high-res. pixel 0, inds_fine adds 0 ... rat2 - 1
    inds_out = rat2 * m_inds_nest + inds_fine
    # Calculate angles
    th_out, ph_out = hp.pix2ang(nside_out, inds_out, nest=True, lonlat=False)
    # Note: for rat2 = 16384 and 150 PSs, the chance that there exists at least a pair of equal th/ph_out is > 50%!
    return th_out, ph_out


# @ray.remote
def run(flux_arr, indices_ROI, temp, exp, pdf_psf_sampler, name="map", save=False, upscale_nside=16384,
        verbose=False, is_nest=False):
    """
    Runs point source Monte Carlo by reading in template, source count distribution parameters, exposure
    map, and the user defined PSF.
    :param flux_arr: array of fluxes of sources to generate
    :param indices_ROI: indices of the ROI that shall be kept
    :param temp: HEALPix numpy array of template
    :param exp: HEALPix numpy array of exposure map
    :param pdf_psf_sampler: user-defined PSF: object of type PDFSampler (see class above). Can be None: no PSF.
    :param name: string for the name of output .npy file
    :param save: option to save map to .npy file
    :param upscale_nside: nside to use for randomly determining the PS location *within* each pixel
    :param verbose: print when starting and finishing
    :param is_nest: if True: temp and exp are assumed to be in NEST format instead of RING (output: same format!)
    :returns: HEALPix format numpy array of simulated map (if getnopsf == True: shape: npix x 2, else: npix),
              list of counts for each PS in the ROI
    """

    # Generate simulated counts map
    map_arr = make_map(flux_arr, indices_ROI, temp, exp, pdf_psf_sampler, upscale_nside,
                                                verbose, is_nest)

    # Save the file as an .npy file
    if save:
        np.save(str(name) + ".npy", np.array(map_arr).astype(np.int32))

    if verbose:
        print("Done simulation.")

    map_arr_return = np.array(map_arr).astype(np.int32)
    return map_arr_return


def make_map(flux_arr, indices_ROI, temp, exp, pdf_psf_sampler, upscale_nside=16384, verbose=False, is_nest=False):
    """
    Given an array of fluxes for each source, template & exposure map, and user defined PSF, simulates and returns
      1) a counts map,
      2) the counts map before the PSF was applied,
      3) the list of the photons per PS (len: # PSs, sum: # counts in the map)
    The source positions are determined by
      1) drawing from a multinomial distribution as determined by the template
      2) randomly determining the PS position within each pixel for each PS
    :param flux_arr: list/array for source fluxes
    :param indices_ROI: indices of the ROI
    :param temp: array of template: MUST BE NORMALISED TO SUM UP TO UNITY!
    :param exp: array of exposure map
    :param pdf_psf_sampler: user-defined PSF: object of type PDFSampler (see class above). Can be None (no PSF).
    :param upscale_nside: nside to use for randomly determining the PS location *within* each pixel
    :param verbose: print when starting
    :param is_nest: if True: temp and exp are assumed to be in NEST format instead of RING (output: same format!)
    :returns: array of simulated counts map, list of counts for each PS
    """
    exp.setflags(write=1)  # these numpy arrays may be read-only! need to change this for ray!
    temp.setflags(write=1)
    if type(flux_arr) == list:
        flux_arr = np.asarray(flux_arr)
    flux_arr.setflags(write=1)

    N = len(flux_arr)  # number of PSs
    NPIX = len(temp)
    NSIDE = hp.npix2nside(NPIX)

    # Initialise the map
    map_arr = np.zeros(NPIX, dtype=np.int32)
    map_arr_no_PSF = np.zeros(NPIX, dtype=np.int32)

    if verbose:
        print("Simulating counts maps ...")

    # Check that template is normalised such that it sums up to unity.
    # Do NOT allow to normalise within this function because of parallelisation (could change template in memory)
    assert(np.abs(np.sum(temp) - 1) < 1e-5), "Template is not normalised!"

    # Draw pixel positions of the PSs and get an array with the (possibly multiple) indices
    inds_PS_bool = stats.multinomial.rvs(p=temp, n=N)  # boolean array: inds_PS_bool.sum() == N
    inds_PS = np.repeat(np.arange(NPIX), inds_PS_bool)  # array with indices: len(inds_PS) == N

    # If no PSs: return at this point
    if len(inds_PS) == 0:
        return map_arr, map_arr_no_PSF, []

    # Randomly shift the PS centres by upscaling to higher nside and randomly moving the centres to those at higher res:
    # this means that for e.g. nside = 128 and upscale_nside = 16384, there are upscale_npix / npix = 16384 possible
    # locations for each PS within each nside = 128 pixel.
    # Note that the HEALPix tesselation is equal area -> (discretised) uniform PDF within each pixel.
    th, ph = random_u_grade_ang(inds_PS, nside_in=NSIDE, nside_out=upscale_nside, is_nest=is_nest)

    # Determine the pixels corresponding to (th, ph):
    # should not have changed since we're only moving the PS within each pixel!
    pix_PS = hp.ang2pix(NSIDE, th, ph, nest=is_nest)
    assert np.all(pix_PS == inds_PS), "The PSs are not in the correct pixels anymore!"

    # Find expected number of source photons and then do a Poisson draw.
    # Weight the total flux by the expected flux in that bin
    num_phot = np.random.poisson(flux_arr * exp[pix_PS])

    # Get array containing the pixel of each count before applying the PSF
    pix_counts = np.repeat(pix_PS, num_phot)

    # if no PSF:
    if pdf_psf_sampler is None:
        posit = pix_counts

    # PSF correction:
    else:
        # Create a rotation matrix for each source.
        # Shift phi coord pi/2 to correspond to center of HEALPix map during rotation.
        phm = ph + np.pi / 2.

        # Each source is initially treated as being located at theta=0,phi=0 as
        # the drawn PSF distances simply corresponds to photon theta position.
        # A random phi value [0,2pi] is then drawn. Each photon is then rotated
        # about the x axis an angle corresponding to the true theta position of
        # the source, followed by a rotation about the z axis by the true phi
        # position plus an additional pi/2 radians.
        a0 = np.zeros(N)
        a1 = np.ones(N)
        rotx = np.array([[a1, a0, a0], [a0, np.cos(th), -np.sin(th)], [a0, np.sin(th), np.cos(th)]])
        rotz = np.array([[np.cos(phm), -np.sin(phm), a0], [np.sin(phm), np.cos(phm), a0], [a0, a0, a1]])

        # Sample distances from PSF for each source photon.
        n_counts_tot = num_phot.sum()
        dist_flat = pdf_psf_sampler(n_counts_tot)  # list of distances for flattened counts, len: n_counts_tot
        assert len(dist_flat) == n_counts_tot

        # Reshape: 3 x 3 x N  ->  N x 3 x 3, then tile: one matrix for each count -> num_phot x 3 x 3
        rotx_tiled = np.repeat(np.transpose(rotx, [2, 0, 1]), num_phot, axis=0)
        rotz_tiled = np.repeat(np.transpose(rotz, [2, 0, 1]), num_phot, axis=0)

        # Draw random phi positions for all the counts from [0, 2pi]
        randPhi = 2 * np.pi * np.random.uniform(0.0, 1.0, n_counts_tot)

        # Convert the theta and phi to x,y,z coords.
        X = np.array(hp.ang2vec(dist_flat, randPhi))

        # Rotate coords over the x axis (first: make X a matrix for each count)
        Xp = rotx_tiled @ np.expand_dims(X, -1)

        # Rotate again, over the z axis (then: remove the additional dimension to get a 3d-vector for each count)
        Xp = np.squeeze(rotz_tiled @ Xp, -1)

        # Determine pixel location from x,y,z values.
        posit = hp.vec2pix(NSIDE, Xp[:, 0], Xp[:, 1], Xp[:, 2], nest=is_nest)


    # Add all the counts: note: use "at" such that multiple counts in a pixel are added
    np.add.at(map_arr, posit, int(1))  # pixels AFTER PSF
    np.add.at(map_arr_no_PSF, pix_counts, int(1))  # pixels BEFORE PSF

    # Restrict to ROI
    map_arr = map_arr[indices_ROI]

    # Return map
    return map_arr