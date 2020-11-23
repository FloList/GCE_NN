"""
This generates GCE maps for the SCD test.
"""
import numpy as np
import healpy as hp
import os
import pickle
from collections import defaultdict
import ray
import psutil
import time
import sys
import random
from gce_utils import mkdir_p, get_template, auto_garbage_collect
from NPTFit import create_mask as cm  # Module for creating masks
######################################################
# Running on supercomputer?
if "flo" in os.path.expanduser("~"):
    HPC = False
else:
    HPC = True
######################################################
if HPC:
    JOB_ID = sys.argv[1]  # when running on supercomputer: provide the job ID as an input parameter
else:
    JOB_ID = 0
print("JOB ID is", JOB_ID, ".\n")

######################################
start_time = time.time()

# Settings
fermi_folder = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data' if HPC else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'
output_path = '/scratch/u95/fl9575/GCE_v2/data/GCE_maps_SCD' if HPC else "/home/flo/PycharmProjects/GCE/data/GCE_maps_SCD"
outer_ring = 30.0  # rather choose this larger: maps can still be confined to a smaller region (and sources can be masked) when combining the models
max_NP_sources = None  # max. number of NP sources per template (to reduce computation time): must be None for SCD test!
CONST_EXP = True  # if True: use constant exposure (mean of Fermi exposure, for experiments)
if CONST_EXP:
    output_path += "_const_exp"

name = "GCE_maps"  # Name of output file
nside = 128
npix = hp.nside2npix(nside)
nchunk = int(100) if HPC else int(2)  # Number of chunks to compute
nsim_per_chunk = int(250) if HPC else int(10)  # Number of simulations per chunk and per model (one file per chunk)
which_SCD = "bright"  # bright, def-bright, default, def-dim, dim, OR: no_PS (yields almost no_PS)
gce_dm_on = False  # Add GCE

all_models_P = ["iso", "dif", "bub", "gce", "gce_12", "dif_O_pibs", "dif_O_ic", "gce_12_N", "gce_12_S"]
all_models_NP = ["iso_PS", "disk_PS", "thin_disk_PS", "thick_disk_PS", "gce_PS", "gce_12_PS", "bub_PS"]

#############
# CHOOSE MODELS
T_P = []  # ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
T_NP = ["gce_12_PS"]
#############

# Define priors for parameter estimation (ROUGHLY: see Leane Table S1)
prior_dict = defaultdict()

# Poissonian templates: A
prior_dict["iso"] = [-0.3, -0.3]
prior_dict["dif_O_pibs"] = [0.91, 0.91]
prior_dict["dif_O_ic"] = [0.58, 0.58]
prior_dict["bub"] = [-0.06, -0.06]
if gce_dm_on:
    prior_dict["gce_12"] = [0.032, 0.032]  # DM with the same no. of counts as GCE PS (~12150)
else:
    prior_dict["gce_12"] = [-1.49, -1.49]  # (almost) no DM

# Non-Poissonian templates: A, n_1, .., n_k, S_1, ... S_{k-1}
if which_SCD is "bright":
    prior_dict["gce_12_PS"] = [[-3.13, -3.13], [10, 10], [-1.2, -1.2], [57.5720, 57.5720]]  # BRIGHT
elif which_SCD is "def-bright":
    prior_dict["gce_12_PS"] = [[-2.31, -2.31], [10, 10], [-1.2, -1.2], [22.3981, 22.3981]]  # DEF - BRIGHT
elif which_SCD is "default":
    prior_dict["gce_12_PS"] = [[-1.49, -1.49], [10, 10], [-1.2, -1.2], [8.7139, 8.7139]]  # DEFAULT
elif which_SCD is "def-dim":
    prior_dict["gce_12_PS"] = [[-0.67, -0.67], [10, 10], [-1.2, -1.2], [3.3901, 3.3901]]  # DEF - DIM
elif which_SCD is "dim":
    prior_dict["gce_12_PS"] = [[0.15, 0.15], [10, 10], [-1.2, -1.2], [1.3189, 1.3189]]  # DIM
elif which_SCD is "no_PS":
    prior_dict["gce_12_PS"] = [[-5.5, -5.5], [10, 10], [-1.2, -1.2], [8.7139, 8.7139]]  # (almost) NO PS
else:
    raise NotImplementedError

# Exposure map
exp = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
mean_exp = np.mean(exp)
if CONST_EXP:
    exp_orig = exp
    exp = np.ones_like(exp_orig) * mean_exp

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

# Set up the mask for the ROI
total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=outer_ring)

# Initialise the dictionaries that will be used
data_out = defaultdict()

# Create the output folder (if it doesn't exist yet)
mkdir_p(output_path)

# Print
print("Starting map generation for '{0}'.".format(name))
print("Number of chunks: {0}, number of simulations per chunk: {1}\n -> {2} maps per model.".format(nchunk, nsim_per_chunk, nchunk * nsim_per_chunk))
print("Max. number of NP sources per template: {0}.".format(max_NP_sources))

# For the parameter randomness: use random module since numpy number generator behaves weird when running on multiple processors
# Start with the Poissonian models
for temp in T_P:
    print("Starting with Poissonian model", temp)
    T = get_template(fermi_folder, temp)
    if CONST_EXP:
        T *= exp / exp_orig

    # Make a subfolder
    if gce_dm_on and temp == "gce_12":
        temp_folder = os.path.join(output_path, temp, "gce_dm_on")
    elif not gce_dm_on and temp == "gce_12":
        temp_folder = os.path.join(output_path, temp, "gce_dm_off")
    else:
        temp_folder = os.path.join(output_path, temp)
    mkdir_p(temp_folder)

    # For each chunk
    for chunk in range(nchunk):

        # Draw the log amplitude
        A = np.asarray([random.uniform(prior_dict[temp][0], prior_dict[temp][1]) for _ in range(nsim_per_chunk)])

        # Generate the maps
        # EXPOSURE IS ALREADY INCORPORATED IN TEMPLATES!
        sim_maps = np.asarray([np.random.poisson((10.0 ** A[i]) * T * (1 - total_mask_neg)) for i in range(nsim_per_chunk)]).T

        # Save settings
        if chunk == 0 and int(JOB_ID) == 0:
            settings_out = dict()
            settings_out["T"] = T
            settings_out["priors"] = prior_dict[temp]
            settings_out["exp"] = exp
            settings_out["rescale"] = rescale
            settings_out["unmasked_pix"] = np.argwhere(1 - total_mask_neg).flatten()
            print("Writing settings file...")
            with open(os.path.join(temp_folder, name + "_settings.pickle"), 'wb') as f:
                pickle.dump(settings_out, f)

        # Save maps
        # The full map can be recovered as map_full = np.zeros(npix), map_full[data_out["unmasked_pix"]] = data_out["val"]
        data_out["data"] = (sim_maps[(1 - total_mask_neg).astype(bool), :])
        data_out["info"] = dict()
        data_out["info"]["A"] = A
        with open(os.path.join(temp_folder, name + "_" + str(JOB_ID) + "_" + str(chunk) + ".pickle"), 'wb') as f:
            pickle.dump(data_out, f)

# Initialise Ray
if T_NP:
    if HPC:
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus, memory=2000000000, object_store_memory=1000000000)  # memory, object_store_memory can be set
    else:
        num_cpus = 2
        ray.init(num_cpus=num_cpus)
    print("Running on", num_cpus, "CPUs.")

    # Define a function for the simulation of the Non-Poissonian models
    @ray.remote
    def create_simulated_map(n_, F_, A_, T_, exp_, psf_r_, name_, max_NP_sources_):
        if not HPC and '/home/flo/PycharmProjects/GCE/NPTFit-Sim/NPTFit-Sim' not in sys.path:
            sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim")
            sys.path.append("/home/flo/PycharmProjects/GCE/NPTFit-Sim/NPTFit-Sim")
        import ps_mc
        return ps_mc.run(n_, F_, A_, T_, exp_, psf_r_,
                         max_NP_sources=max_NP_sources_, name=name_, save=False)[0]

# Now, do the Non-Poissonian models
for temp in T_NP:
    print("Starting with Non-Poissonian model", temp)
    T = get_template(fermi_folder, temp[:-3])
    if CONST_EXP:
        T_corr = T * exp / exp_orig
    else:
        T_corr = T / rescale

    # Make a subfolder
    temp_folder = os.path.join(output_path, temp, which_SCD)
    mkdir_p(temp_folder)

    # Put the large arrays to the object store
    exp_ID = ray.put(exp)
    T_corr_masked_ID = ray.put(T_corr * (1 - total_mask_neg))

    # For each chunk
    for chunk in range(nchunk):

        # Draw the parameters
        temp_len = len(prior_dict[temp])
        A = np.asarray([random.uniform(prior_dict[temp][0][0], prior_dict[temp][0][1]) for _ in range(nsim_per_chunk)])
        n = np.asarray([[random.uniform(prior_dict[temp][i][0], prior_dict[temp][i][1]) for _ in range(nsim_per_chunk)]
                        for i in range(1, temp_len // 2 + 1)]).T
        S = np.asarray([[random.uniform(prior_dict[temp][i][0], prior_dict[temp][i][1]) for _ in range(nsim_per_chunk)]
                        for i in range(temp_len // 2 + 1, temp_len)]).T

        # Sort count breaks
        S = np.fliplr(np.sort(S, axis=1))

        # For NP models: counts -> flux by dividing by mean_exp
        F = S / mean_exp
        A_corr = A + cor_term

        # Run the simulations
        indices_for_sim = np.arange(nsim_per_chunk)
        indices_for_sim_old = np.arange(nsim_per_chunk)
        sim_maps = np.zeros((len(T_corr), nsim_per_chunk))
        while len(indices_for_sim) > 0:
            finished_sims = np.asarray(ray.get([create_simulated_map.remote(n[i, :], F[i], A_corr[i],
                                                                            T_corr_masked_ID,
                                                                            exp_ID, psf_r, "map_" + temp, max_NP_sources
                                                                            ) for i in indices_for_sim])).T

            # # # # #
            # import seaborn as sns
            # import matplotlib.pyplot as plt
            # finished_sims[:, 0].max()
            # hp.orthview(np.log10(1 + finished_sims[:, 0]), cmap="rocket")
            # fig = plt.gcf()
            # fig.savefig(which_SCD + "logplot.pdf", dpi=600)
            # # # # #

            # Indices that need to be simulated again because too many NP sources
            indices_for_sim = indices_for_sim[np.argwhere(finished_sims.sum(0) < 0).flatten()]

            indices_too_few_counts = []

            indices_done = np.setdiff1d(indices_for_sim_old, indices_for_sim)
            indices_done_loc = np.argwhere(finished_sims.sum(0) >= 0).flatten()

            sim_maps[:, indices_done] = finished_sims[:, indices_done_loc]
            indices_for_sim_old = indices_for_sim

            # Reduce A by 1 if too many sources
            if len(indices_for_sim) > 0:
                A[np.setdiff1d(indices_for_sim, indices_too_few_counts)] -= 1.0
                A_corr[np.setdiff1d(indices_for_sim, indices_too_few_counts)] -= 1.0
                if np.any(A[np.setdiff1d(indices_for_sim, indices_too_few_counts)] < prior_dict[temp][0][0]):
                    print("WARNING! 'A' was reduced to be smaller than lower prior limit!")
            else:
                break

            # Collect garbage
            auto_garbage_collect()

        # Save settings
        if chunk == 0 and int(JOB_ID) == 0:
            settings_out = dict()
            settings_out["T"] = T
            settings_out["T_corr"] = T_corr
            settings_out["priors"] = prior_dict[temp]
            settings_out["exp"] = exp
            settings_out["rescale"] = rescale
            settings_out["max_NP_sources"] = max_NP_sources
            settings_out["unmasked_pix"] = np.argwhere(1 - total_mask_neg).flatten()
            print("Writing settings file...")
            with open(os.path.join(temp_folder, name + "_settings.pickle"), 'wb') as f:
                pickle.dump(settings_out, f)

        # Save maps
        # The full map can be recovered as map_full = np.zeros(npix), map_full[data_out["unmasked_pix"]] = data_out["val"]
        data_out["data"] = (sim_maps[(1 - total_mask_neg).astype(bool), :])
        data_out["info"] = dict()
        data_out["info"]["A"] = A
        data_out["info"]["A_corr"] = A_corr
        data_out["info"]["n"] = n
        data_out["info"]["S"] = S
        data_out["info"]["F"] = F
        with open(os.path.join(temp_folder, name + "_" + str(JOB_ID) + "_" + str(chunk) + ".pickle"),
                  'wb') as f:
            pickle.dump(data_out, f)

print("Done! Computation took {0} seconds.".format(time.time() - start_time))
# Loading pickle file e.g.: data = pickle.load( open( "./data/GCE_smooth_and_PS/GCE_masked.pickle", "rb" ) )
