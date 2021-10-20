"""
This generates maps for each PS template and saves map for each template separately, using the python-only MC code.
Then, they can be combined arbitrarily for the creation of training data.
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
from gce_utils import mkdir_p, get_template, auto_garbage_collect, backup_one_file, multipage, masked_to_full, \
    get_is_HPC, get_fermi_folder_basename, get_Fermi_PDF_sampler
######################################################
HPC = get_is_HPC()
######################################################
if not HPC:
    new_lib = '/usr/local/lib'
    try:
        if new_lib not in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] += ':'+new_lib
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
nside = 256# if HPC else 128
this_name = 'GCE_maps_573w'

if HPC == 0:
    output_path = os.path.join('/home/flo/PycharmProjects/GCE/data', this_name)
elif HPC == 1:
    output_path = os.path.join('/scratch/u95/fl9575/GCE_v2/data', this_name)
elif HPC == 2:
    output_path = os.path.join('/scratch/dss/GCE_NN/data', this_name)
output_path += "_" + str(nside)

outer_ring = 25.0  # choose EXACTLY the same mask and exposure that will be used for the NN training!
inner_band = 2.0
MASK_TYPE = "3FGL"  # mask for known bright PSs: either: "None", "3FGL", "4FGL"
max_total_flux = 3e-07  # 3e-7 corresponds to ~25,000 counts
Poisson_A_is_log = False  # is A for Poisson templates specified as log10(A)?
CONST_EXP = False
do_Fermi_PSF = True  # if True: use Fermi PSF for PS templates, if False: no PSF!
name = "GCE_maps_dNdF"  # Name of output file
npix = hp.nside2npix(nside)
nchunk = int(300) if HPC else int(2)  # Number of chunks to compute
nsim_per_chunk = int(250) if HPC else int(250)  # Number of simulations per chunk and per model (one file per chunk)
fermi_folder = get_fermi_folder_basename(HPC, w573=True)
fermi_folder += "/fermi_data_" + str(nside)
save_example_plot = True
n_plot = 5
output_NEST = True  # if True: save the output in NEST format, otherwise RING
#############
# CHOOSE MODELS
T_P = ["iso", "dif", "dif_O_pibs", "dif_O_ic", "dif_A_pibs", "dif_A_ic", "dif_F_pibs", "dif_F_ic", "bub", "bub_var"]
# T_P = ["dif_O_pibs", "dif_O_ic"]
T_NP = ["gce_12_PS", "thin_disk_PS", "thick_disk_PS"]
# T_NP = []
#############

prior_dict = dict()

# Poissonian templates: A (use try_out_Poisson_priors.py to check the resulting no. of counts)
for key in ["iso", "bub", "bub_var"]:
    prior_dict[key] = [0, 2]
for key in ["gce", "gce_12"]:
    prior_dict[key] = [0, 2.5]
for key in ["dif_A_pibs", "dif_F_pibs", "dif_O_pibs"]:
    prior_dict[key] = [7, 14]
for key in ["dif_A_ic", "dif_F_ic", "dif_O_ic"]:
    prior_dict[key] = [4, 9]
prior_dict["dif"] = [8, 18]  # p6v11 containing both contributions (pibs + ic)

# For larger nsides: need to scale Poissonian priors by npix ratio
if nside != 128:
    for key in prior_dict:
        prior_dict[key][0] /= (nside // 128) ** 2
        prior_dict[key][1] /= (nside // 128) ** 2

# Define priors for parameter estimation (for skew Gaussian distributions)
# prior_dict["iso_PS"] = {"mean_exp": [-12, -9], "var_exp": 0.25, "skew_std": 3.0, "flux_lims": [-7.819, -7.041], "flux_log": True}  # flux limits correspond to 1,000 - 6,000 counts, logarithmic
prior_dict["gce_12_PS"] = prior_dict["thin_disk_PS"] = prior_dict["thick_disk_PS"] \
   = {"mean_exp": [-12, -9], "var_exp": 0.25, "skew_std": 3.0, "flux_lims": [0, 1.4e-07], "flux_log": False}  # 0 - 12,500 for 573w data, uniform
# NOTE: THE UPPER FLUX LIMITS IS --> HALF <--- OF THE MAX. EXPECTED GCE: TWO TEMPLATE MAPS WILL BE SUMMED UP!

# Exposure map
fermi_exp = get_template(fermi_folder, "exp")
if output_NEST:
    fermi_exp = hp.reorder(fermi_exp, r2n=True)
mean_exp = np.mean(fermi_exp)
fermi_rescale = fermi_exp / mean_exp

if CONST_EXP:
    exp = np.ones_like(fermi_exp) * mean_exp
else:
    exp = fermi_exp

cor_term = np.log10(mean_exp)
rescale = exp / mean_exp

# Set a random seed for numpy (using random, because numpy duplicates random number generator for multiple processes)
random_seed = random.randint(0, int(2**32 - 1))
np.random.seed(random_seed)
print("Job ID:", JOB_ID, "Random Seed:", random_seed)

# PSF: use Fermi-LAT PSF
if do_Fermi_PSF:
    pdf = get_Fermi_PDF_sampler()
else:
    pdf = None

# Set up the mask for the ROI
total_mask_neg = cm.make_mask_total(band_mask=True, band_mask_range=inner_band,
                                    mask_ring=True, inner=0, outer=outer_ring, nside=nside)

if MASK_TYPE == "3FGL":
    total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_folder, "3FGL_mask"))).astype(bool)
elif MASK_TYPE == "4FGL":
    total_mask_neg = (1 - (1 - total_mask_neg) * (1 - get_template(fermi_folder, "4FGL_mask"))).astype(bool)

if output_NEST:
    total_mask_neg = hp.reorder(total_mask_neg, r2n=True)

# Initialise the dictionaries that will be used
data_out = defaultdict()

# Create the output folder (if it doesn't exist yet)
mkdir_p(output_path)

# Print
print("Starting map generation for '{0}'.".format(name))
print("Number of chunks: {0}, number of simulations per chunk: "
      "{1}\n -> {2} maps per model.".format(nchunk, nsim_per_chunk, nchunk * nsim_per_chunk))

# Back up this file
if JOB_ID == 0:
    backup_folder = os.path.join(output_path, "src_backup")
    mkdir_p(backup_folder)
    backup_one_file(__file__, backup_folder)

# Start with the Poissonian models
for temp in T_P:
    print("Starting with Poissonian model", temp)
    T = get_template(fermi_folder, temp)
    if output_NEST:
        T = hp.reorder(T, r2n=True)
    if CONST_EXP:
        T /= fermi_rescale

    # Get pixels that are not masked
    unmasked_pix = np.argwhere(1 - total_mask_neg).flatten()

    # Mask template and compress
    T_masked_compressed = (T * (1 - total_mask_neg))[unmasked_pix]

    # Make a subfolder
    temp_folder = os.path.join(output_path, temp)
    mkdir_p(temp_folder)

    # For each chunk
    for chunk in range(nchunk):

        # Draw the (log) amplitude
        A = np.asarray([random.uniform(prior_dict[temp][0], prior_dict[temp][1]) for _ in range(nsim_per_chunk)])

        # Generate the maps
        # EXPOSURE IS ALREADY INCORPORATED IN TEMPLATES!
        if Poisson_A_is_log:
            sim_maps = np.asarray([np.random.poisson((10.0 ** A[i]) * T_masked_compressed) for i in range(nsim_per_chunk)])
        else:
            sim_maps = np.asarray([np.random.poisson(A[i] * T_masked_compressed) for i in range(nsim_per_chunk)])

        # Save settings
        if chunk == 0 and int(JOB_ID) == 0:
            settings_out = dict()
            settings_out["T"] = T
            settings_out["priors"] = prior_dict[temp]
            settings_out["is_log_A"] = Poisson_A_is_log
            settings_out["exp"] = exp
            settings_out["rescale"] = rescale
            settings_out["unmasked_pix"] = unmasked_pix
            settings_out["format"] = "NEST" if output_NEST else "RING"
            settings_out["mask_type"] = MASK_TYPE
            settings_out["outer_ring"] = outer_ring
            settings_out["inner_band"] = inner_band
            settings_out["nside"] = nside
            print("Writing settings file...")
            with open(os.path.join(temp_folder, name + "_settings.pickle"), 'wb') as f:
                pickle.dump(settings_out, f)

        # Save maps
        # The full map can be recovered as map_full = np.zeros(npix), map_full[data_out["unmasked_pix"]] = data_out["val"]
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
            hp.mollview(total_mask_neg, title="Mask (" + str(MASK_TYPE) + ")", nest=output_NEST)
            for i in range(n_plot):
                hp.mollview(masked_to_full(sim_maps[i, :], unmasked_pix, nside=nside),
                            title=int(np.round(sim_maps[i, :].sum())), nest=output_NEST)

            multipage(os.path.join(output_path, temp + "_examples.pdf"))
            plt.close("all")


# Initialise Ray
if T_NP:
    if HPC == 1:
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus, memory=2000000000, object_store_memory=2000000000)  # memory, object_store_memory can be set
    elif HPC == 2:
        num_cpus = psutil.cpu_count(logical=False)
        ray.init(num_cpus=num_cpus, include_dashboard=False, object_store_memory=2000000000)
    else:
        num_cpus = 4
        ray.init(num_cpus=num_cpus)  # , memory=500000000, object_store_memory=500000000)
    print("Running on", num_cpus, "CPUs.")

    # Put the large array / objects that are template-independent into the object store
    exp_ID = ray.put(exp)
    pdf_ID = ray.put(pdf)

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


# Do the Non-Poissonian models
for temp in T_NP:
    print("Starting with Non-Poissonian model", temp)
    T = get_template(fermi_folder, temp[:-3])
    if output_NEST:
        T = hp.reorder(T, r2n=True)
    T_corr = T / fermi_rescale

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
    temp_folder = os.path.join(output_path, temp)
    mkdir_p(temp_folder)

    # Put the large arrays / objects to the object store
    T_corr_final_ID = ray.put(T_corr_final)

    # For each chunk
    for chunk in range(nchunk):
        print("Starting with chunk", chunk)

        # Draw the parameters
        mean_draw = np.random.uniform(*prior_dict[temp]["mean_exp"], size=nsim_per_chunk)
        var_draw = prior_dict[temp]["var_exp"] * np.random.chisquare(1, size=nsim_per_chunk)
        skew_draw = np.random.normal(loc=0, scale=prior_dict[temp]["skew_std"], size=nsim_per_chunk)

        # DEBUG
        # sim_maps, n_phot, flux_arr = create_simulated_map(skew_draw[0], mean_draw[0], np.sqrt(var_draw[0]), prior_dict[temp]["flux_lims"],
        # T_corr_final, exp, pdf, "map_" + temp, flux_log=prior_dict[temp]["flux_log"])

        sim_maps, n_phot, flux_arr = map(list, zip(*ray.get(
            [create_simulated_map.remote(skew_draw[i_PS], mean_draw[i_PS], np.sqrt(var_draw[i_PS]),
             prior_dict[temp]["flux_lims"], T_corr_final_ID, exp_ID, pdf_ID, "map_" + temp,
             flux_log=prior_dict[temp]["flux_log"]) for i_PS in range(nsim_per_chunk)])))

        sim_maps = np.asarray(sim_maps) * np.expand_dims((1 - total_mask_neg)[None], -1)   # apply mask again: counts might have leaked outside due to PSF
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
            settings_out["priors"] = prior_dict[temp]
            settings_out["exp"] = exp  # exposure
            settings_out["rescale"] = rescale
            settings_out["max_NP_sources"] = np.nan  # not set here
            settings_out["unmasked_pix"] = np.argwhere(1 - total_mask_neg).flatten()
            settings_out["format"] = "NEST" if output_NEST else "RING"
            settings_out["mask_type"] = MASK_TYPE
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
            hp.mollview(total_mask_neg, title="Mask (" + str(MASK_TYPE) + ")", nest=output_NEST)
            for i in range(n_plot):
                hp.mollview(sim_maps[i, :, 0], title=int(np.round(sim_maps[i, :, 0].sum())), nest=output_NEST)

            multipage(os.path.join(output_path, temp + "_examples.pdf"))
            plt.close("all")


print("Done! Computation took {0} seconds.".format(time.time() - start_time))
# Loading pickle file e.g.: data = pickle.load( open( "./data/GCE_smooth_and_PS/GCE_masked.pickle", "rb" ) )
