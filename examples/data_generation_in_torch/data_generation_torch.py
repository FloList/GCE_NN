from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import torch
from healpy_torch import pix2ang, ang2pix, ring2nest, nest2ring, ang2vec, vec2ang


class PDFSampler:
    def __init__(self, xvals, pofx, device="cpu"):
        """ At outset sort and calculate CDF so not redone at each call
        :param xvals: tensor of x values
        :param pofx: tensor of associated p(x) values (does not need to be normalised)
        :param device: device to use
        """
        self.xvals = xvals
        self.pofx = pofx
        self.device = device

        # Check p(x) >= 0 for all x, otherwise stop
        assert(torch.all(pofx >= 0)), "pdf cannot be negative"

        # Sort values by their p(x) value, for more accurate sampling
        self.sortxvals = torch.argsort(self.pofx)
        self.pofx = self.pofx[self.sortxvals]

        # Calculate cdf
        self.cdf = torch.cumsum(self.pofx, dim=0)

    def __call__(self, samples):
        """
        When class called returns samples number of draws from pdf
        :param samples: number of draws you want from the pdf
        :returns: number of random draws from the provided PDF
        """
        # Random draw from a uniform, up to max of the cdf, which need
        # not be 1 as the pdf does not have to be normalised
        unidraw = torch.rand(size=(samples.item(),), device=self.device) * self.cdf[-1]
        cdfdraw = torch.searchsorted(self.cdf, unidraw)
        cdfdraw = self.sortxvals[cdfdraw]
        return self.xvals[cdfdraw]



# Random upscaling:
# PS positions are determined in 2 steps:
#   1. Draw from a multinomial distribution where the probabilities are defined by the template to determine the pixel
#   2. Shift each PS to the centre of a randomly determined high-res. pixel contained within the original low-res. pixel
#      (otherwise, all PSs would be located at the centre of the low-res. pixel)
# The following function takes care of step 2.
def random_u_grade_ang(m_inds, nside_in=0, nside_out=16384):
    """
    Random upscaling of the PS positions, given the pixel centres at resolution nside_in.
    Each PS is moved to one of the high-resolution pixel centres at resolution nside_out.
    For example, for nside_in = 128 and nside_out = 16384, one has npix_out / npix_in = 16384
    possible PS locations within each pixel.
    NOTE: indices must correspond to NEST format!
    :param m_inds: indices of the PSs w.r.t. nside_in, in NEST ordering
    :param nside_in: nside in
    :param nside_out: nside to use for upscaling
    :return: theta, phi of randomly placed PSs within each pixel
    """
    if m_inds.shape[0] == 0:
        return m_inds
    n_ps = m_inds.shape[0]  # number of point sources
    hp.isnsideok(nside_out, nest=True)  # check that nside_out is okay
    npix_in = hp.nside2npix(nside_in)
    npix_out = hp.nside2npix(nside_out)
    rat2 = npix_out // npix_in
    # For each PS, draw a random fine pixel within the coarse pixel
    inds_fine = torch.randint(0, rat2, size=(n_ps,), device=m_inds.device)
    # Set indices w.r.t. upscaled nside (NEST): rat2 * m_inds_nest -> high-res. pixel 0, inds_fine adds 0 ... rat2 - 1
    inds_out = rat2 * m_inds + inds_fine
    # Calculate angles
    th_ph = pix2ang(nest2ring(nside_out, inds_out), nside_out)
    th_out, ph_out = th_ph[:, 0], th_ph[:, 1]
    # Note: for rat2 = 16384 and 150 PSs, the chance that there exists at least a pair of equal th/ph_out is > 50%!
    m_inds_out = ring2nest(nside, ang2pix(nside, th_out, ph_out))
    assert torch.all(m_inds_out == m_inds), "PSs have been shifted outside the low res. pixel - this shouldn't happen!"
    return th_out, ph_out


def make_map(flux_arr, temp, exp, pdf_psf_sampler, upscale_nside=16384, device="cpu"):
    """
    Given an array of fluxes for each source, template & exposure map, and user defined PSF, simulates and returns
      1) a counts map,
      2) the counts map before the PSF was applied,
      3) the list of the photons per PS (len: # PSs, sum: # counts in the map)
    The source positions are determined by
      1) drawing from a multinomial distribution as determined by the template
      2) randomly determining the PS position within each pixel for each PS
    NOTE: Everything needs to be provided in NEST format!
    :param flux_arr: list/array for source fluxes
    :param temp: array of template: MUST BE NORMALISED TO SUM UP TO UNITY!
    :param exp: array of exposure map
    :param pdf_psf_sampler: user-defined PSF: object of type PDFSampler (see class above). Can be None (no PSF).
    :param upscale_nside: nside to use for randomly determining the PS location *within* each pixel
    :param device: device to use
    :returns: tensor of simulated counts map, the same map before applying the PSF, list of counts for each PS,
              flux array

    """
    n = flux_arr.shape[0]  # number of PSs
    npix = temp.shape[0]
    nside = hp.npix2nside(npix)

    # Initialise the map
    map_arr = torch.zeros(npix, dtype=torch.int64, device=device)
    map_arr_no_psf = torch.zeros(npix, dtype=torch.int64, device=device)

    # Check that template is normalised such that it sums up to unity.
    assert(torch.abs(torch.sum(temp) - 1) < 1e-5), "Template is not normalised!"

    # Draw pixel positions of the PSs and get an array with the (possibly multiple) indices
    inds_ps = torch.multinomial(temp, n, replacement=True)  # indices of the PSs w.r.t. nside, in NEST ordering
    ps_per_pix = torch.bincount(inds_ps, minlength=npix)  # number of PSs per pixel
    inds_repeated = torch.repeat_interleave(torch.arange(npix, device=device), ps_per_pix)  # array with indices: len(inds_PS) == n

    # If no PSs: return at this point
    if inds_ps.shape[0] == 0:
        return map_arr, map_arr_no_psf, [], []

    # Randomly shift the PS centres by upscaling to higher nside and randomly moving the centres to those at higher res:
    # this means that for e.g. nside = 128 and upscale_nside = 16384, there are upscale_npix / npix = 16384 possible
    # locations for each PS within each nside = 128 pixel.
    # Note that the HEALPix tesselation is equal area -> (discretised) uniform PDF within each pixel.
    th, ph = random_u_grade_ang(inds_repeated, nside_in=nside, nside_out=upscale_nside)

    # Determine the pixels corresponding to (th, ph):
    # should not have changed since we're only moving the PS within each pixel!
    pix_ps_ring = ang2pix(nside, th, ph)
    pix_ps = ring2nest(nside, pix_ps_ring)
    # assert torch.all(pix_ps == inds_repeated), "The PSs are not in the correct pixels anymore!"

    # Find expected number of source photons and then do a Poisson draw.
    # Weight the total flux by the exposure map.
    num_phot = torch.poisson(flux_arr * exp[pix_ps]).long()

    # Get array containing the pixel of each count before applying the PSF
    pix_counts = torch.repeat_interleave(pix_ps, num_phot)

    # if no PSF:
    if pdf_psf_sampler is None:
        posit = pix_counts

    # PSF correction:
    else:
        # Create a rotation matrix for each source.
        # Shift phi coord pi/2 to correspond to center of HEALPix map during rotation.
        phm = ph + torch.pi / 2.

        # Each source is initially treated as being located at theta=0, phi=0 as
        # the drawn PSF distances simply correspond to photon theta position.
        # A random phi value [0, 2pi] is then drawn. Each photon is then rotated
        # about the x axis an angle corresponding to the true theta position of
        # the source, followed by a rotation about the z axis by the true phi
        # position plus an additional pi/2 radians.
        a0 = torch.zeros(n, device=device)
        a1 = torch.ones(n, device=device)

        rotx = torch.stack([torch.stack([a1, a0, a0]),
                            torch.stack([a0, torch.cos(th), -torch.sin(th)]),
                            torch.stack([a0, torch.sin(th), torch.cos(th)])])

        rotz = torch.stack([torch.stack([torch.cos(phm), -torch.sin(phm), a0]),
                            torch.stack([torch.sin(phm), torch.cos(phm), a0]),
                            torch.stack([a0, a0, a1])])

        # Sample distances from PSF for each source photon.
        n_counts_tot = num_phot.sum()
        dist_flat = pdf_psf_sampler(n_counts_tot)  # list of distances for flattened counts, len: n_counts_tot
        assert len(dist_flat) == n_counts_tot

        # Reshape: 3 x 3 x N  ->  N x 3 x 3, then tile: one matrix for each count -> num_phot x 3 x 3
        rotx_tiled = torch.repeat_interleave(rotx.permute(2, 0, 1), num_phot, dim=0)
        rotz_tiled = torch.repeat_interleave(rotz.permute(2, 0, 1), num_phot, dim=0)

        # Draw random phi positions for all the counts from [0, 2pi]
        rand_phi = 2 * torch.pi * torch.rand(size=(n_counts_tot.item(),), device=device)

        # Convert the theta and phi to x,y,z coords.
        x = ang2vec(dist_flat, rand_phi)

        # Rotate coords over the x axis (first: make x a matrix for each count)
        xp = rotx_tiled @ torch.unsqueeze(x, -1)

        # Rotate again, over the z axis (then: remove the additional dimension to get a 3d-vector for each count)
        xp = torch.squeeze(rotz_tiled @ xp, -1)

        # Determine pixel location from x,y,z values (vector -> angle -> RING pixel -> NEST pixel)
        ang = vec2ang(xp)
        posit_r = ang2pix(nside, ang[:, 0], ang[:, 1])
        posit = ring2nest(nside, posit_r)

    # Add the counts to the map
    map_arr = torch.scatter_add(map_arr, 0, posit, torch.ones_like(posit))  # after PSF
    map_arr_no_psf = torch.scatter_add(map_arr_no_psf, 0, pix_counts, torch.ones_like(pix_counts))  # before PSF

    # Return map, map before PSF, and num_phot_cleaned
    return map_arr, map_arr_no_psf, num_phot, flux_arr


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Define resolution parameters
    nside = 64
    npix = hp.nside2npix(nside)
    upscale_nside = 16384

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define a template and the exposure map
    # Set up everything with RING ordering
    temp_r = np.zeros(npix)
    temp_r[:npix // 2] = 10.0  # Value in the northern hemisphere
    temp_r[npix // 2:] = 1.0  # Value in the southern hemisphere
    temp_r /= np.sum(temp_r)  # Normalise to unity
    temp_r = torch.tensor(temp_r, device=device)
    exp_r = torch.ones(npix, device=device)

    # Convert to NEST ordering
    r2n_inds = nest2ring(nside, torch.arange(npix))
    temp = temp_r[r2n_inds]
    exp = exp_r[r2n_inds]

    # PSF
    # pdf_psf_sampler = None
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
            return 1. / (2. * torch.pi * sigma ** 2.) * (1. - 1. / gamma) * (
                    1. + (x ** 2. / (2. * gamma * sigma ** 2.))) ** (
                -gamma)

        def fermi_psf_inner(r_):
            return fcore * king_fn(r_ / spe, score, gcore) + (1 - fcore) * king_fn(r_ / spe, stail, gtail)

        return fermi_psf_inner(r)


    def get_fermi_pdf_sampler(n_points_f=int(1e6)):
        f = torch.linspace(0, torch.pi, n_points_f, device=device)
        pdf_psf = f * fermi_psf(f)
        pdf = PDFSampler(f, pdf_psf, device=device)
        return pdf

    pdf_psf_sampler = get_fermi_pdf_sampler()

    # Flux array
    flux_arr = torch.ones(10000, device=device) * 100.0

    # Make map
    m, m_no_psf, n_phot, f_arr = make_map(flux_arr, temp, exp, pdf_psf_sampler, upscale_nside=upscale_nside,
                                          device=device)
    m_np = m.cpu().numpy()
    m_no_psf_np = m_no_psf.cpu().numpy()
    exp_np = exp.cpu().numpy()
    temp_np = temp.cpu().numpy()

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(24, 23))
    hp.mollview(m_np, nest=True, fig=fig, sub=(2, 3, 1), title="Counts map")
    hp.mollview(m_no_psf_np, nest=True, fig=fig, sub=(2, 3, 2), title="Counts map before PSF")
    hp.mollview(m_np - m_no_psf_np, nest=True, fig=fig, sub=(2, 3, 3), title="Difference", min=-10, max=10,
                cmap="seismic")
    hp.mollview(temp_np, nest=True, fig=fig, sub=(2, 3, 4), title="Template")
    hp.mollview(exp_np, nest=True, fig=fig, sub=(2, 3, 5), title="Exposure")
    plt.show()

    # Delete axes
    for ax in axs.flatten():
        ax.axis("off")


