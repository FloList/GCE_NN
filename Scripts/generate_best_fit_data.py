"""
This generates samples with best fit parameters for the GCE as determined by fit_Fermi_counts.py
"""
import numpy as np
import healpy as hp
import os
import pickle
from collections import defaultdict
import ray
import psutil
from gce_utils import get_template, mkdir_p
import time
import sys
from NPTFit import create_mask as cm  # Module for creating masks
######################################################
# Running on supercomputer?
if "flo" in os.path.expanduser("~"):
    HPC = False
else:
    HPC = True
######################################################
M = True                    # mask 3FGL?
TASK = "GCE_for_letter"     # "GCE_for_letter", "GCE_for_letter_thick", ...
######################################
if HPC:
    JOB_ID = sys.argv[1]  # when running on supercomputer: provide the job ID as an input parameter
else:
    JOB_ID = 0
print("JOB IS is", JOB_ID, ".\n")
######################################

start_time = time.time()

# Settings
fermi_folder = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data' if HPC else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'
output_path = '/scratch/u95/fl9575/GCE_v2/data/Best_fit' if HPC else "/home/flo/PycharmProjects/GCE/data/Best_fit"
output_path += "_" + TASK
outer_ring_min = 15.0
outer_ring_max = 25.0
n_rings = 64
max_NP_sources = None  # max. number of NP sources per template (to reduce computation time)

name = "Best_fit_masked" if M else "Best_fit_unmasked"  # Name of output file
nside = 128
npix = hp.nside2npix(nside)
nsim = int(250) if HPC else int(10)  # Number of simulations per chunk (one file per chunk)

# Templates
all_models_P = ["iso", "dif", "bub", "gce", "gce_12", "dif_O_pibs", "dif_O_ic"]
all_models_NP = ["iso_PS", "disk_PS", "gce_PS", "gce_12_PS", "bub_PS", "thin_disk_PS", "thick_disk_PS"]

#############
# CHOOSE MODELS
if TASK == "GCE_for_letter":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
    T_NP = ["thin_disk_PS", "gce_12_PS"]
elif TASK == "GCE_for_letter_thick":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
    T_NP = ["thick_disk_PS", "gce_12_PS"]
else:
    raise NotImplementedError
#############

# Exposure map
exp = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
mean_exp = np.mean(exp)
cor_term = np.log10(mean_exp)
rescale = exp / mean_exp

# PSF: use Fermi-LAT PSF
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

def Fermi_PSF(r):
    return fcore * king_fn(r / spe, score, gcore) + (1 - fcore) * king_fn(r / spe, stail, gtail)

# Lambda function to pass user defined PSF, includes Jacobian factor
psf_r = lambda r: Fermi_PSF(r)

# alternatively: Gaussian approximation
# sigma_gaussian = 0.1812 * np.pi / 180.
# psf_gaussian = lambda r: np.exp(-r ** 2. / (2. * sigma_gaussian ** 2.))

# Set up the masks
total_mask_neg = np.asarray([cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=ring) for ring in np.linspace(outer_ring_min, outer_ring_max, n_rings)])
if M:
    pscmask = np.load(os.path.join(fermi_folder, "fermidata_pscmask.npy"))
    total_mask_neg = (1 - (1 - total_mask_neg) * (1 - pscmask[None]))

# Initialise Ray
if HPC:
    num_cpus = psutil.cpu_count(logical=True)
    ray.init(num_cpus=num_cpus, memory=2000000000, object_store_memory=1000000000)  # memory, object_store_memory can be set
else:
    num_cpus = 4
    ray.init(num_cpus=num_cpus)
print("Running on", num_cpus, "CPUs.")

# Initialise the dictionaries that will be used
sim_map_dict = defaultdict()
total_flux_dict = defaultdict()
flux_fraction_dict = defaultdict()

# Load the templates
T_dict, T_corr_dict = dict(), dict()
for temp in T_P:
    T_dict[temp] = get_template(fermi_folder, temp)
for temp in T_NP:
    poiss_temp = get_template(fermi_folder, temp[:-3])
    T_dict[temp[:-3]] = T_dict[temp] = poiss_temp

# Set the best-fit values (see the results from fit_Fermi_counts.py, check that masking, templates, & ROI are the same)
A_dict, A_corr_dict, n_dict, S_dict, F_dict = dict(), dict(), dict(), dict(), dict()
# NOTE: flux breaks must be given from highest to lowest!
if M:
    if TASK == "GCE_for_letter":
        # Median values within 25*
        #A_dict["iso"] = -0.43
        #A_dict["bub"] = -0.06
        #A_dict["gce_12"] = -1.57
        #A_dict["dif_O_pibs"] = 0.90
        #A_dict["dif_O_ic"] = 0.58
        #A_dict["gce_12_PS"] = -0.71
        #A_dict["thin_disk_PS"] = -1.47
        #n_dict["gce_12_PS"] = np.asarray([33.97, -0.93])
        #n_dict["thin_disk_PS"] = np.asarray([3.32, -0.97])
        #S_dict["gce_12_PS"] = np.asarray([3.75])
        #S_dict["thin_disk_PS"] = np.asarray([4.71])

        # Median values within 20*
        #A_dict["iso"] = -0.37
        #A_dict["bub"] = -0.07
        #A_dict["gce_12"] = -1.71
        #A_dict["dif_O_pibs"] = 0.91
        #A_dict["dif_O_ic"] = 0.55
        #A_dict["gce_12_PS"] = -0.77
        #A_dict["thin_disk_PS"] = -1.28
        #n_dict["gce_12_PS"] = np.asarray([34.23, -0.93])
        #n_dict["thin_disk_PS"] = np.asarray([3.24, -0.95])
        #S_dict["gce_12_PS"] = np.asarray([4.28])
        #S_dict["thin_disk_PS"] = np.asarray([3.94])

        # Median values within 15*
        #A_dict["iso"] = -0.49
        #A_dict["bub"] = -0.20
        #A_dict["gce_12"] = -1.74
        #A_dict["dif_O_pibs"] = 0.92
        #A_dict["dif_O_ic"] = 0.62
        #A_dict["gce_12_PS"] = -1.15
        #A_dict["thin_disk_PS"] = -4.49
        #n_dict["gce_12_PS"] = np.asarray([33.89, -1.05])
        #n_dict["thin_disk_PS"] = np.asarray([31.47, -1.12])
        #S_dict["gce_12_PS"] = np.asarray([7.13])
        #S_dict["thin_disk_PS"] = np.asarray([19.15])

        # Median values within 10*
        A_dict["iso"] = -1.15
        A_dict["bub"] = -0.90
        A_dict["gce_12"] = -1.84
        A_dict["dif_O_pibs"] = 0.94
        A_dict["dif_O_ic"] = 0.66
        A_dict["gce_12_PS"] = -1.46
        A_dict["thin_disk_PS"] = -4.50
        n_dict["gce_12_PS"] = np.asarray([34.34, -1.08])
        n_dict["thin_disk_PS"] = np.asarray([31.58, -1.13])
        S_dict["gce_12_PS"] = np.asarray([9.52])
        S_dict["thin_disk_PS"] = np.asarray([21.71])

    elif TASK == "GCE_for_letter_thick":
        # Median values
        A_dict["iso"] = -1.34
        A_dict["bub"] = -0.03
        A_dict["gce_12"] = -1.81
        A_dict["dif_O_pibs"] = 0.91
        A_dict["dif_O_ic"] = 0.64
        A_dict["gce_12_PS"] = -1.02
        A_dict["thick_disk_PS"] = -4.23
        n_dict["gce_12_PS"] = np.asarray([34.73, -1.19])
        n_dict["thick_disk_PS"] = np.asarray([2.73, -0.86])
        S_dict["gce_12_PS"] = np.asarray([6.05])
        S_dict["thick_disk_PS"] = np.asarray([28.60])

else:
    raise NotImplementedError

# For NP models: count -> flux by dividing by mean exp
for temp in T_NP:
    F_dict[temp] = S_dict[temp] / mean_exp
    A_corr_dict[temp] = A_dict[temp] + cor_term
    T_corr_dict[temp] = T_dict[temp] / rescale

# Simulate the Poissonian models
# EXPOSURE IS ALREADY INCORPORATED IN TEMPLATES!
for temp in T_P:
    # NOTE: take the LARGEST mask
    sim_map_dict[temp] = np.asarray([np.random.poisson((10.0 ** A_dict[temp]) * T_dict[temp] * (1 - total_mask_neg[-1, :])) for i in range(nsim)]).T

# Plot the total Poissonian counts
# plot_ind = 0
# hp.mollview(np.asarray([sim_map_dict[temp][:, plot_ind] for temp in T_P]).sum(0))

# Simulate the Non-Poissonian models: THIS TAKES TIME!
@ray.remote
def create_simulated_map(n_, F_, A_, T_, exp_, psf_r_, name_, max_NP_sources_):
    if not HPC and '/home/flo/PycharmProjects/GCE/NPTFit-Sim' not in sys.path:
        sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim")
        sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim/NPTFit-Sim")
    else:
        sys.path = [dir.replace("PS-Sim", "NPTFit-Sim") for dir in sys.path]
    import ps_mc
    return ps_mc.run(n_, F_, A_, T_, exp_, psf_r_, max_NP_sources=max_NP_sources_, name=name_, save=False)

for temp in T_NP:
    # NOTE: take the LARGEST mask
    if not HPC and '/home/flo/PycharmProjects/GCE/NPTFit-Sim' not in sys.path:
        sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim")
        sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim/NPTFit-Sim")
    else:
        sys.path = [dir.replace("PS-Sim", "NPTFit-Sim") for dir in sys.path]
    # import ps_mc
    # ps_mc.run(n_dict[temp], F_dict[temp], A_corr_dict[temp], T_corr_dict[temp] * (1 - total_mask_neg[-1, :]),
    #                                                                          exp, psf_r, "map_" + temp, max_NP_sources)
    sim_map_dict[temp] = np.asarray(ray.get([create_simulated_map.remote(n_dict[temp], F_dict[temp],
                                                                             A_corr_dict[temp], T_corr_dict[temp] * (1 - total_mask_neg[-1, :]),
                                                                             exp, psf_r, "map_" + temp, max_NP_sources
                                                                             ) for i in range(nsim)])).T

# Make a plot for each template
# [hp.mollview(sim_map_dict[temp][:, 0]) for temp in T_NP]

# Calculate total counts
total_sim_map = np.asarray([v for k, v in sim_map_dict.items()]).sum(0)

# Calculate flux fractions for each radius
for temp in T_P + T_NP:
    total_flux_dict[temp] = np.asarray([(sim_map_dict[temp] / np.expand_dims(exp, -1) * np.expand_dims(1 - total_mask_neg[rad, :], -1)).sum(0) for rad in range(total_mask_neg.shape[0])])
total_flux = np.asarray([v for k, v in total_flux_dict.items()]).sum(0)
for temp in T_P + T_NP:
    flux_fraction_dict[temp] = total_flux_dict[temp] / total_flux


# Print some stats
print("Templates:")
print(list(flux_fraction_dict.keys()))
print("Radius of ROI between", outer_ring_min, "and", outer_ring_max)
print("Max. flux fraction for each template:")
print([np.round(flux_fraction_dict[key].max(), 2) for key in flux_fraction_dict.keys()])
print("Min. flux fraction for each template:")
print([np.round(flux_fraction_dict[key].min(), 2) for key in flux_fraction_dict.keys()])
print("Mean flux fraction for each template:")
print([np.round(flux_fraction_dict[key].mean(), 2) for key in flux_fraction_dict.keys()])
print("Median flux fraction for each template:")
print([np.round(np.median(flux_fraction_dict[key]), 2) for key in flux_fraction_dict.keys()])
print("Avg. total number of counts:")
print(np.mean(total_sim_map.sum(0)))

# Only store what is important:
# NOTE: OUTPUT IS IN TERMS OF COUNTS! FOR NN: REMOVE EXPOSURE CORRECTION IN PREPROCESSING!
mkdir_p(output_path)

data_out = dict()
settings_out = dict()
settings_out["T"] = T_dict
settings_out["T_corr"] = T_corr_dict
settings_out["exp"] = exp
settings_out["rescale"] = rescale
print("Writing settings file...")
settings_out["unmasked_pix"] = np.argwhere(1 - total_mask_neg[-1, :]).flatten()  # reduce to the largest ROI
with open(os.path.join(output_path, name + "_settings.pickle"), 'wb') as f:
    pickle.dump(settings_out, f)


data_out["data"] = (total_sim_map[(1 - total_mask_neg[-1, :]).astype(bool), :])
data_out["flux_fraction"] = flux_fraction_dict
data_out["info"] = dict()
data_out["info"]["A"] = A_dict
data_out["info"]["A_corr"] = A_corr_dict
data_out["info"]["n"] = n_dict
data_out["info"]["S"] = S_dict
data_out["info"]["F"] = F_dict
data_out["info"]["total_flux"] = total_flux_dict

with open(os.path.join(output_path, name + "_" + str(JOB_ID) + ".pickle"), 'wb') as f:
    pickle.dump(data_out, f)

print("Computation finished.\nTime elapsed:", time.time() - start_time, 's.')

