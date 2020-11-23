# If the PS is given by a singly broken power law:
# the total flux should be given by
#     F_tot^PS = A_p S_b^2 / E_p (1 / (n_1 - 2) + 1 / (2 - n_2).
# This script validates this formula numerically. See also Lee et al. 2016 Supp. Mat.
# (https://link.aps.org/doi/10.1103/PhysRevLett.116.051103).

import sys
sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim")
sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim/NPTFit-Sim")
import ps_mc
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os

# Select task
TASK = 0  # 0: vary A and S_b simultaneously, 1: vary n_1 and n_2 simultaneously

# Global settings
nside = 16
n_samples = 30
npix = hp.nside2npix(nside)
temp = 0.1 * np.ones(npix).astype(float)
EXP = np.ones(npix).astype(float)
mean_exp = np.mean(EXP)
# Define parameters that specify the PSF
sigma = 2.00 * np.pi / 180.
# Lambda function to pass user defined PSF
psf_r = lambda r: np.exp(-r ** 2. / (2. * sigma ** 2.))

# Check scaling of A and S_b
if TASK == 0:
    n_SCD = np.array([15.00, -1.0])  # indices for SCD
    S_array = np.sqrt(2 ** np.arange(8, 0, -1))
    A_array = np.log10(1e-2 * 2 ** (np.arange(0, 8)))
    print((S_array ** 2) * (10 ** A_array))
    samples = range(n_samples)
    counts = np.zeros((len(A_array), len(samples)))

    for i, (S, A) in enumerate(zip(S_array, A_array)):
        # Calculate flux break from counts break and mean exposure
        F = np.array([S]) / mean_exp

        # Convert log-normalization term, A, into terms of flux
        A = A + np.log10(mean_exp)

        # Simulate
        for sample in samples:
            counts[i, sample] = ps_mc.run(n_SCD, F, A, temp, EXP, psf_r, name="map", save=False)[0].sum()

    plt.errorbar(x=range(counts.shape[0]), y=counts.mean(1), yerr=counts.std(1))
    plt.title(r"Varying $A$ and $S$", size=14)
    plt.xlabel("Parameter combinations")
    plt.ylabel("Total number of counts (exposure map is constant)")
    plt.ylim(0, 500)

# Check scaling of n_1 and n_2
elif TASK == 1:
    S = np.array([10.00])  # SCD break, here in terms of counts
    A = -1.00  # log-normalization
    samples = range(n_samples)
    n_1_array = np.arange(30, 10, -4)
    C = 0.5
    n_2_array = (1 / (1 / (n_1_array - 2) - C)) + 2
    print(1 / (n_1_array - 2) + 1 / (2 - n_2_array))
    counts = np.zeros((len(n_1_array), len(samples)))

    for i, (n_1, n_2) in enumerate(zip(n_1_array, n_2_array)):
        n_SCD = np.array([n_1, n_2])  # indices for SCD
        name = "simulation"  # Name of output file

        # Calculate flux break from counts break and mean exposure
        F = S / mean_exp

        # Convert log-normalization term, A, into terms of flux
        A = A + np.log10(mean_exp)

        # Simulate
        for sample in samples:
            counts[i, sample] = ps_mc.run(n_SCD, F, A, temp, EXP, psf_r, name="map", save=False)[0].sum()

    plt.errorbar(x=range(counts.shape[0]), y=counts.mean(1), yerr=counts.std(1))
    plt.title(r"Varying $n_1$ and $n_2$", size=14)
    plt.xlabel("Parameter combinations")
    plt.ylabel("Total number of counts (exposure map is constant)")
    plt.ylim(0, 1800)
