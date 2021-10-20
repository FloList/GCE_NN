"""
Generate maps for the NS mismodelling example: iso NP + iso P template, and save map for each template separately.
Then, the maps can be combined for the creation of training data.
"""
import random
import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict
import ray
import psutil
import time
import sys
from gce_utils import mkdir_p, auto_garbage_collect, backup_one_file, multipage, masked_to_full, get_is_HPC

######################################################
HPC = get_is_HPC()
######################################################
if not HPC:
    new_lib = '/usr/local/lib'
    try:
        if not new_lib in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] += ':' + new_lib
    except:
        os.environ['LD_LIBRARY_PATH'] = new_lib
    os.chdir('/home/flo/PycharmProjects/GCE/DeepSphere/Fast_PS_generation')
    sys.path.append('/home/flo/PycharmProjects/GCE/DeepSphere/Fast_PS_generation')
from NPTFit import create_mask as cm  # Module for creating masks

######################################################

if HPC:
    JOB_ID = sys.argv[1]
else:
    JOB_ID = 0
print("JOB ID is", JOB_ID, ".\n")

######################################
start_time = time.time()

# Settings
nside = 128
this_name = 'NS_example_maps'

if HPC == 0:
    output_path = os.path.join('/home/flo/PycharmProjects/GCE/data', this_name)
elif HPC == 1:
    output_path = os.path.join('/scratch/u95/fl9575/GCE_v2/data', this_name)
elif HPC == 2:
    output_path = os.path.join('/scratch/dss/GCE_NN/data', this_name)
output_path += "_" + str(nside)

outer_ring = 25.0  # choose EXACTLY the same mask and exposure that will be used for the NN training!
inner_band = 0.0
max_total_flux = 100000
CONST_EXP = True
do_Fermi_PSF = False  # if True: use Fermi PSF for PS templates, if False: no PSF!
name = "NS_maps"  # Name of output file
npix = hp.nside2npix(nside)
nchunk = int(250) if HPC else int(2)  # Number of chunks to compute
nsim_per_chunk = int(250) if HPC else int(100)  # Number of simulations per chunk and per model (one file per chunk)
save_example_plot = True
n_plot = 5
output_NEST = True  # if True: save the output in NEST format, otherwise RING
#############
# CHOOSE MODELS
T_P = ["iso"]
T_NP = ["iso_PS"]
#############

prior_dict = dict()

# Poissonian templates: A
prior_dict["iso"] = [0, 12]
prior_dict["iso_PS"] = {"mean_exp": [0, 1.5], "var_exp": 0.1, "skew_std": 3.0,
                        "flux_lims": [1, 100000], "flux_log": False}

# Exposure map
exp = np.ones(npix)

# Set a random seed for numpy (using random, because numpy duplicates random number generator for multiple processes)
random_seed = random.randint(0, int(2 ** 32 - 1))
np.random.seed(random_seed)
print("Job ID:", JOB_ID, "Random Seed:", random_seed)

# PSF: use Fermi-LAT PSF
pdf = None

# Set up the mask for the ROI
total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band,
                                    mask_ring=True, inner=0, outer=outer_ring, nside=nside)

if output_NEST:
    total_mask_neg = hp.reorder(total_mask_neg, r2n=True)

# Initialise the dictionaries that will be used
data_out = defaultdict()

# Create the output folder (if it doesn't exist yet)
mkdir_p(output_path)

# Print
print("Starting map generation for '{0}'.".format(name))
print("Number of chunks: {0}, number of simulations per chunk: {1}\n -> {2} "
      "maps per model.".format(nchunk, nsim_per_chunk, nchunk * nsim_per_chunk))

# Back up this file
if int(JOB_ID) == 0:
    backup_folder = os.path.join(output_path, "src_backup")
    mkdir_p(backup_folder)
    backup_one_file(__file__, backup_folder)

# Start with the Poissonian model
T = np.ones(npix)
print("Starting with the isotropic Poissonian model")

# Get pixels that are not masked
unmasked_pix = np.argwhere(1 - total_mask_neg).flatten()

# Mask template and compress
T_masked_compressed = (T * (1 - total_mask_neg))[unmasked_pix]

# Make a subfolder
temp_folder = os.path.join(output_path, "iso")
mkdir_p(temp_folder)

# For each chunk
for chunk in range(nchunk):

    # Draw the normalisation
    A = np.asarray([random.uniform(prior_dict["iso"][0], prior_dict["iso"][1]) for _ in range(nsim_per_chunk)])

    # Generate the maps
    sim_maps = np.asarray([np.random.poisson(A[i] * T_masked_compressed) for i in range(nsim_per_chunk)])

    # Save settings
    if chunk == 0 and int(JOB_ID) == 0:
        settings_out = dict()
        settings_out["T"] = T
        settings_out["priors"] = prior_dict["iso"]
        settings_out["is_log_A"] = False
        settings_out["exp"] = exp
        settings_out["rescale"] = exp
        settings_out["unmasked_pix"] = unmasked_pix
        settings_out["format"] = "NEST" if output_NEST else "RING"
        settings_out["mask_type"] = "None"
        settings_out["outer_ring"] = outer_ring
        settings_out["inner_band"] = inner_band
        settings_out["nside"] = nside
        print("Writing settings file...")
        with open(os.path.join(temp_folder, name + "_settings.pickle"), 'wb') as f:
            pickle.dump(settings_out, f)

    # Save maps
    # The full map can be recovered as map_full = np.zeros(npix), map_full[data_out["unmasked_pix"]] = data_out["data"]
    data_out["data"] = sim_maps
    data_out["info"] = dict()
    data_out["info"]["A"] = A
    with open(os.path.join(temp_folder, name + "_" + str(JOB_ID) + "_" + str(chunk) + ".pickle"), 'wb') as f:
        pickle.dump(data_out, f)

    # Plot some maps and save
    if chunk == 0 and int(JOB_ID) == 0 and save_example_plot:
        plt.ioff()
        hp.mollview(T, title="Template (exposure-corrected)", nest=output_NEST)
        hp.mollview(exp, title="Exposure (nside = " + str(nside) + ")", nest=output_NEST)
        hp.mollview(total_mask_neg, title="ROI mask", nest=output_NEST)
        for i in range(n_plot):
            hp.mollview(masked_to_full(sim_maps[i, :], unmasked_pix, nside=nside),
                        title=int(np.round(sim_maps[i, :].sum())), nest=output_NEST)

        multipage(os.path.join(output_path, "iso" + "_examples.pdf"))
        plt.close("all")

# Initialise Ray
if T_NP:
    if HPC == 1:
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus, memory=2000000000,
                 object_store_memory=2000000000)  # memory, object_store_memory can be set
    elif HPC == 2:
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus, include_dashboard=False, object_store_memory=2000000000)
    else:
        num_cpus = 4
        ray.init(num_cpus=num_cpus)  # , memory=500000000, object_store_memory=500000000)
    print("Running on", num_cpus, "CPUs.")

    # Put the large array / objects that are template-independent into the object store
    exp_ID = ray.put(exp)

    # Define a function for the simulation of the Non-Poissonian models
    @ray.remote
    def create_simulated_map(skew_, loc_, scale_, flux_lims_, T_, exp_, pdf_, name_, size_approx_mean=10000,
                             max_iter=50, flux_log=True):
        from ps_mc_fast import run
        total_counts = 0
        iter = 0

        while total_counts <= 0 and iter < max_iter:
            # Draw the desired flux
            if flux_log:
                flux_desired = 10 ** np.random.uniform(*flux_lims_)
            else:
                flux_desired = np.random.uniform(*flux_lims_)
            # Calculate the expected value of 10^X
            exp_value = (10 ** sp.stats.skewnorm.rvs(skew_, loc=loc_, scale=scale_, size=size_approx_mean)).mean()
            # Determine the number of sources
            n_sources = int(np.round(flux_desired / exp_value))
            # Initialise total flux
            tot_flux = np.infty
            # Draw fluxes until total flux is in valid range
            while tot_flux >= max_total_flux:
                flux_arr_ = 10 ** sp.stats.skewnorm.rvs(skew_, loc=loc_, scale=scale_, size=n_sources)
                tot_flux = flux_arr_.sum()
                # If total flux > max-total_flux: reduce n_sources
                if tot_flux > max_total_flux:
                    n_sources = int(max(1, n_sources // 1.5))
            # Do MC run
            map_, n_phot_, = run(np.asarray(flux_arr_), T_, exp_, pdf_, name_, save=False, getnopsf=True, getcts=True,
                                 upscale_nside=16384, verbose=False, is_nest=output_NEST)
            # Check that there are counts in the map
            total_counts = map_[:, 0].sum()
            # Increase iter
            iter += 1
        return map_, n_phot_, flux_arr_

# Do the Non-Poissonian model
print("Starting with isotropic Non-Poissonian model")
T = T_corr = np.ones(npix)

# Apply mask
T_corr_masked = T_corr * (1 - total_mask_neg)

# Template needs to be normalised to sum up to unity for the new implementation!
# Might need to do this twice because of rounding errors
T_corr_final = T_corr_masked / T_corr_masked.sum()
while T_corr_final.sum() > 1.0:
    T_corr_final /= T_corr_final.sum()
if T_corr_final.sum() != 1.0:
    print("WARNING: TEMPLATE SUM IS NOT EXACTLY 1 BUT", T_corr_final.sum(), "!")

# Make a subfolder
temp_folder = os.path.join(output_path, "iso_PS")
mkdir_p(temp_folder)

# Put the large arrays / objects to the object store
T_corr_final_ID = ray.put(T_corr_final)

# For each chunk
for chunk in range(nchunk):
    print("Starting with chunk", chunk)

    # Draw the parameters
    mean_draw = np.random.uniform(*prior_dict["iso_PS"]["mean_exp"], size=nsim_per_chunk)
    var_draw = prior_dict["iso_PS"]["var_exp"] * np.random.chisquare(1, size=nsim_per_chunk)
    skew_draw = np.random.normal(loc=0, scale=prior_dict["iso_PS"]["skew_std"], size=nsim_per_chunk)

    # DEBUG
    # sim_maps, n_phot, flux_arr = create_simulated_map(skew_draw[0], mean_draw[0], np.sqrt(var_draw[0]), prior_dict["iso_PS"]["flux_lims"],
    #                                                   T_corr_final, exp, pdf, "map_iso_PS", flux_log=prior_dict["iso_PS"]["flux_log"])

    sim_maps, n_phot, flux_arr = map(list, zip(*ray.get(
        [create_simulated_map.remote(skew_draw[i_PS], mean_draw[i_PS], np.sqrt(var_draw[i_PS]),
                                     prior_dict["iso_PS"]["flux_lims"], T_corr_final_ID, exp_ID, pdf, "map_iso_PS",
                                     flux_log=prior_dict["iso_PS"]["flux_log"]) for i_PS in range(nsim_per_chunk)])))

    sim_maps = np.asarray(sim_maps) * np.expand_dims((1 - total_mask_neg)[None],
                                                     -1)  # apply mask again: counts might have leaked outside due to PSF
    assert np.all(sim_maps[:, :, 0].sum(1) == [n_phot[i].sum() for i in range(nsim_per_chunk)]), \
        "Photons counts in maps and n_phot lists are not consistent! Aborting..."

    # NOTE: finished_sims_raw[:, :, 0].sum(1) may be < finished_sims_raw[:, :, 1].sum(1):
    # some photons may leak out of the ROI due to PSF and are cut off above

    # Collect garbage
    auto_garbage_collect()

    # Save settings
    if chunk == 0 and int(JOB_ID) == 0:
        settings_out = dict()
        settings_out["T"] = T
        settings_out["T_corr"] = T_corr
        settings_out["priors"] = prior_dict["iso_PS"]
        settings_out["exp"] = exp  # exposure
        settings_out["rescale"] = exp
        settings_out["max_NP_sources"] = np.nan  # not set here
        settings_out["unmasked_pix"] = np.argwhere(1 - total_mask_neg).flatten()
        settings_out["format"] = "NEST" if output_NEST else "RING"
        settings_out["mask_type"] = "None"
        settings_out["outer_ring"] = outer_ring
        settings_out["inner_band"] = inner_band
        settings_out["nside"] = nside
        print("Writing settings file...")
        with open(os.path.join(temp_folder, name + "_settings.pickle"), 'wb') as f:
            pickle.dump(settings_out, f)

    # Save maps
    # The full map can be recovered as map_full = np.zeros(npix), map_full[data_out["unmasked_pix"]] = data_out["val"]
    data_out["data"] = (sim_maps[:, (1 - total_mask_neg).astype(bool), :])
    data_out["n_phot"] = n_phot
    data_out["flux_arr"] = flux_arr
    data_out["info"] = dict()
    data_out["info"]["tot_flux"] = np.asarray([np.sum(f) for f in flux_arr])
    data_out["info"]["means"] = mean_draw
    data_out["info"]["vars"] = var_draw
    data_out["info"]["skew"] = skew_draw

    with open(os.path.join(temp_folder, name + "_" + str(JOB_ID) + "_" + str(chunk) + ".pickle"), 'wb') as f:
        pickle.dump(data_out, f)

    # Plot some maps and save
    if chunk == 0 and int(JOB_ID) == 0 and save_example_plot:
        plt.ioff()
        hp.mollview(T_corr_final, title="Template (exposure removed)", nest=output_NEST)
        hp.mollview(exp, title="Exposure (nside = " + str(nside) + ")", nest=output_NEST)
        hp.mollview(total_mask_neg, title="ROI Mask", nest=output_NEST)
        for i in range(n_plot):
            hp.mollview(sim_maps[i, :, 0], title=int(np.round(sim_maps[i, :, 0].sum())), nest=output_NEST)

        multipage(os.path.join(output_path, "iso_PS_examples.pdf"))
        plt.close("all")

print("Done! Computation took {0} seconds.".format(time.time() - start_time))
# Loading pickle file e.g.: data = pickle.load( open( "./data/.../filename.pickle", "rb" ) )
