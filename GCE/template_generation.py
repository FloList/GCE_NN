import numpy as np
import scipy.integrate as integrate
import healpy as hp
import ray
from NPTFit import create_mask as cm


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
    # np.save(filename + "_gamma_" + str(gamma), template_final)

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


if __name__ == "__main__":
    exp = 1.0 * np.ones(hp.nside2npix(128))
    generate_gce_template(exp, gamma=1.0, r_s=20, d_GC=8.5, nside=128, num_cpus=4, filename="GCE_template")