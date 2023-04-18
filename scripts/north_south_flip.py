from numba import njit
import healpy as hp
import numpy as np

nside = 32
npix = hp.nside2npix(nside)

theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
theta_phi = np.stack((theta, phi), -1)
pi_minus_theta = np.pi - theta
pi_minus_theta_phi = np.stack((pi_minus_theta, phi), -1)

@njit
def build_inds():
    inds_i = []
    inds_j = []
    for i, p1 in enumerate(pi_minus_theta_phi):
        for j, p2 in enumerate(theta_phi):
            if np.sum(np.abs(p1 - p2)) < 1e-10:
                inds_i.append(i)
                inds_j.append(j)
    return inds_i, inds_j

inds_i, inds_j = build_inds()

m = np.arange(npix)
assert np.all(np.asarray(inds_i) == np.arange(npix))
m_flipped = m[inds_j]

hp.mollview(m, nest=True, cmap="Spectral")
hp.mollview(m_flipped, nest=True, cmap="Spectral")