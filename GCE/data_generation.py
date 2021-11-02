"""
Generate and save maps for each template.
"""
import random
import numpy as np
from scipy import stats
import healpy as hp
import matplotlib.pyplot as plt
import os
import pickle
from .data_utils import get_fermi_pdf_sampler, masked_to_full
from .utils import multipage, auto_garbage_collect
import ray
import time
import warnings


def generate_template_maps(params, temp_dict, ray_settings, n_example_plots, job_id=0):
    """
    Generate simulated template maps for each template (output format: NESTED!)
    :param params: DotDict containing the settings (see parameters.py)
    :param temp_dict: DotDict containing the templates
    :param ray_settings: dictionary containing the settings for ray
    :param n_example_plots: number of maps to plot and save for each template (as a quick check)
    :param job_id: if running several jobs for the data generation: ID of the current job
    """
    start_time = time.time()

    # Get settings that will be stored in a separate file together with the maps
    t_p = params.mod["models_P"]
    t_ps = params.mod["models_PS"]

    nside = params.data["nside"]
    outer_rad = params.data["outer_rad"]
    inner_band = params.data["inner_band"]
    mask_type = params.data["mask_type"]
    do_fermi_psf = params.data["psf"]
    leakage_delta = params.data["leakage_delta"] if do_fermi_psf else 0

    if "db" in params.keys():
        do_poisson_scatter_p = False if params.db["deactivate_poiss_scatter_for_P"] else True
    else:
        do_poisson_scatter_p = True

    name = params.tt["filename_base"]
    n_chunk = params.tt["n_chunk"]
    n_sim_per_chunk = params.tt["n_sim_per_chunk"]
    poisson_a_is_log = params.tt["poisson_A_is_log"]
    add_two_temps_ps = params.tt["add_two_temps_PS"]

    output_path = params.gen["template_maps_folder"]
    prior_dict = params.tt.priors
    save_example_plot = n_example_plots > 0

    exp = temp_dict["exp"]
    rescale_compressed = temp_dict["rescale_compressed"]

    # Set output dtypes
    dtype_data = np.uint32 if do_poisson_scatter_p else np.float32  # without Poisson draw, counts are non-integer
    dtype_flux_arr = np.float32

    # Set a random seed for numpy (using random because numpy duplicates random number generator for multiple processes)
    random_seed = random.randint(0, int(2 ** 32 - 1))
    np.random.seed(random_seed)
    print("Job ID:", job_id, "Random Seed:", random_seed)

    # PSF: use Fermi-LAT PSF
    if do_fermi_psf:
        pdf = get_fermi_pdf_sampler()
    else:
        pdf = None

    # Get the masks
    total_mask_neg = temp_dict["mask_ROI_full"]  # uncompressed, nest format, contains PS mask if desired
    total_mask_neg_safety = temp_dict["mask_safety_full"]  # the same for the slightly larger ROI

    # Initialise the output dictionary
    data_out = dict()

    # Create the output folder (if it doesn't exist yet)
    os.makedirs(output_path, exist_ok=True)

    # Print
    print("Starting map generation for '{0}'.".format(params.tt["data_name"]))
    print("Number of chunks: {0}, number of simulations per chunk: "
          "{1}\n -> {2} maps per model.".format(n_chunk, n_sim_per_chunk, n_chunk * n_sim_per_chunk))
    if len(add_two_temps_ps) > 0:
        print(" Twice as many maps will be created for", add_two_temps_ps)

    # Start with the Poissonian models
    for temp in t_p:
        print("Starting with Poissonian model '{:}'".format(temp))
        t = temp_dict["T_counts"][temp]  # exposure-corrected template in counts space

        # Get pixels that are not masked
        indices_roi = temp_dict["indices_roi"]

        # Mask template and compress
        t_masked = t * (1 - total_mask_neg)
        t_masked_compressed = t_masked[indices_roi]

        # Make a subfolder
        temp_folder = os.path.join(output_path, temp)
        os.makedirs(temp_folder, exist_ok=True)

        # For each chunk
        for chunk in range(n_chunk):

            # Draw the (log) amplitude
            a = np.asarray([random.uniform(prior_dict[temp][0], prior_dict[temp][1])
                            for _ in range(n_sim_per_chunk)])

            # Generate the maps: NOTE: exposure-correction is included in the Poissonian templates ("T_counts")
            random_draw_fn = np.random.poisson if do_poisson_scatter_p else lambda x: x
            if poisson_a_is_log:
                sim_maps = np.asarray([random_draw_fn((10.0 ** a[i]) * t_masked_compressed)
                                       for i in range(n_sim_per_chunk)])
            else:
                sim_maps = np.asarray([random_draw_fn(a[i] * t_masked_compressed)
                                       for i in range(n_sim_per_chunk)])

            # Save settings
            if chunk == 0 and int(job_id) == 0:
                settings_out = dict()
                settings_out["T"] = t
                settings_out["priors"] = prior_dict[temp]
                settings_out["is_log_A"] = poisson_a_is_log
                settings_out["exp"] = exp
                settings_out["rescale_compressed"] = rescale_compressed
                settings_out["indices_roi"] = indices_roi
                settings_out["format"] = "NEST"
                settings_out["mask_type"] = mask_type
                settings_out["outer_rad"] = outer_rad
                settings_out["inner_band"] = inner_band
                settings_out["leakage_delta"] = leakage_delta
                settings_out["nside"] = nside
                print("   Writing settings file...")
                with open(os.path.join(temp_folder, name + "_settings.pickle"), 'wb') as f:
                    pickle.dump(settings_out, f)

            # Save maps
            # The full map can be recovered as
            # map_full = np.zeros(npix), map_full[data_out["indices_roi"]] = data_out["val"]
            data_out["data"] = sim_maps.astype(dtype_data)
            data_out["info"] = dict()
            data_out["info"]["A"] = a
            with open(os.path.join(temp_folder, name + "_" + str(job_id) + "_" + str(chunk) + ".pickle"), 'wb') as f:
                pickle.dump(data_out, f)

            # Plot some maps and save
            if chunk == 0 and int(job_id) == 0 and save_example_plot:
                plt.ioff()
                hp.mollview(t_masked, title="Template (exposure-corrected)", nest=True)
                hp.mollview(exp, title="Exposure (nside = " + str(nside) + ")", nest=True)
                hp.mollview(total_mask_neg, title="Mask (" + str(mask_type) + ")", nest=True)
                for i in range(n_example_plots):
                    hp.mollview(masked_to_full(sim_maps[i, :], indices_roi, nside=nside),
                                title=int(np.round(sim_maps[i, :].sum())), nest=True)

                multipage(os.path.join(output_path, temp + "_examples.pdf"))
                plt.close("all")

    # Initialise Ray
    if t_ps:
        ray.init(**ray_settings)
        if "num_cpus" in ray_settings.keys():
            print("Ray: running on", ray_settings["num_cpus"], "CPUs.")

        # Put the large array / objects that are template-independent into the object store
        exp_id = ray.put(exp)
        pdf_id = ray.put(pdf)

        # Define a function for the simulation of the point-source models
        @ray.remote
        def create_simulated_map(skew_, loc_, scale_, flux_lims_, enforce_upper_flux_, t_, exp_, pdf_, name_,
                                 inds_outside_roi_, size_approx_mean_=10000, flux_log_=False):
            from .ps_mc import run
            assert np.all(np.isfinite(flux_lims_)), "Flux limits must be finite!"
            max_total_flux = flux_lims_[1] if enforce_upper_flux_ else -np.infty

            # Draw the desired flux
            if flux_log_:
                flux_desired = 10 ** np.random.uniform(*flux_lims_)
            else:
                flux_desired = np.random.uniform(*flux_lims_)
            # Calculate the expected value of 10^X
            exp_value = (10 ** stats.skewnorm.rvs(skew_, loc=loc_, scale=scale_, size=int(size_approx_mean_))).mean()
            # Determine the expected number of sources
            n_sources_exp = flux_desired / exp_value
            # Draw the observed number of sources from a Poisson distribution
            n_sources = np.random.poisson(n_sources_exp)
            # Initialise total flux
            tot_flux = np.infty
            # Draw fluxes until total flux is in valid range
            flux_arr_ = []
            while tot_flux >= max_total_flux:
                flux_arr_ = 10 ** stats.skewnorm.rvs(skew_, loc=loc_, scale=scale_, size=n_sources)
                tot_flux = flux_arr_.sum()
                if not enforce_upper_flux_:
                    break
                # If total flux > max-total_flux: reduce n_sources
                if tot_flux > max_total_flux:
                    n_sources = int(max(1, int(n_sources // 1.05)))

            # Do MC run
            map_, n_phot_, flux_arr_out = run(np.asarray(flux_arr_), t_, exp_, pdf_, name_, save=False, getnopsf=True,
                                              getcts=True, upscale_nside=16384, verbose=False, is_nest=True,
                                              inds_outside_roi=inds_outside_roi_, clean_count_list=False)

            return map_, n_phot_, flux_arr_out

        # Do the point-source models
        for temp in t_ps:
            print("Starting with point-source model '{:}'".format(temp))
            t = temp_dict["T_flux"][temp]  # for point-sources: template after REMOVING the exposure correction is used

            # Apply slightly larger mask
            t_masked = t * (1 - total_mask_neg_safety)

            # Correct flux limit priors for larger mask (after simulating the counts, ROI mask will be applied)
            flux_corr_fac = t_masked.sum() / (t * (1 - total_mask_neg)).sum()
            flux_lims_corr = [None] * 2
            for i in range(2):
                if prior_dict[temp]["flux_log"]:
                    flux_lims_corr[i] = prior_dict[temp]["flux_lims"][i] + np.log10(flux_corr_fac)
                else:
                    flux_lims_corr[i] = prior_dict[temp]["flux_lims"][i] * flux_corr_fac

            # Get indices where PSs are sampled although they lie outside ROI
            inds_ps_outside_roi = set(np.setdiff1d(temp_dict["indices_safety"], temp_dict["indices_roi"]))

            # Template needs to be normalised to sum up to unity for the new implementation!
            # Might need to do this twice because of rounding errors
            t_final = t_masked / t_masked.sum()
            while t_final.sum() > 1.0:
                t_final /= t_final.sum()
            if t_final.sum() != 1.0:
                warnings.warn("Template sum is not exactly 1, but {:}!".format(t_final.sum()))

            # Make a subfolder
            temp_folder = os.path.join(output_path, temp)
            os.makedirs(temp_folder, exist_ok=True)

            # Put the large arrays / objects to the object store
            t_final_id = ray.put(t_final)
            inds_ps_outside_roi_id = ray.put(inds_ps_outside_roi)

            # For each chunk
            this_n_chunk = 2 * n_chunk if temp in add_two_temps_ps else n_chunk
            for chunk in range(this_n_chunk):
                print("  Starting with chunk", chunk)

                # Draw the parameters
                mean_draw = np.random.uniform(*prior_dict[temp]["mean_exp"], size=n_sim_per_chunk)
                var_draw = prior_dict[temp]["var_exp"] * np.random.chisquare(1, size=n_sim_per_chunk)
                skew_draw = np.random.normal(loc=0, scale=prior_dict[temp]["skew_std"], size=n_sim_per_chunk)

                # This code is for debugging without ray
                # sim_maps, n_phot, flux_arr = create_simulated_map(skew_draw[0], mean_draw[0], np.sqrt(var_draw[0]),
                #                                                   flux_lims_corr,
                #                                                   prior_dict[temp]["enforce_upper_flux"],
                #                                                   t_final, exp, pdf, "map_" + temp,
                #                                                   flux_log_=prior_dict[temp]["flux_log"],
                #                                                   inds_outside_roi_=inds_ps_outside_roi)

                sim_maps, n_phot, flux_arr = map(list, zip(*ray.get(
                    [create_simulated_map.remote(skew_draw[i_PS], mean_draw[i_PS], np.sqrt(var_draw[i_PS]),
                                                 flux_lims_corr, prior_dict[temp]["enforce_upper_flux"],
                                                 t_final_id, exp_id, pdf_id, "map_" + temp,
                                                 flux_log_=prior_dict[temp]["flux_log"],
                                                 inds_outside_roi_=inds_ps_outside_roi_id)
                     for i_PS in range(n_sim_per_chunk)])))

                # Apply ROI mask again and cut off counts outside ROI
                sim_maps = np.asarray(sim_maps) * np.expand_dims((1 - total_mask_neg), [0, -1])

                # The following assert is for the scenario where there is NO leakage INTO the ROI, and counts leaking
                # OUT OF the ROI are deleted from photon-count list n_phot
                # assert np.all(sim_maps[:, :, 0].sum(1) == [n_phot[i].sum() for i in range(n_sim_per_chunk)]), \
                #         "Photons counts in maps and n_phot lists are not consistent! Aborting..."

                # The following assert is for the scenario where there is leakage INTO and OUT OF the ROI, and n_phot
                # contains ALL the counts (and only those counts) from PSs within the ROI.
                assert np.all(sim_maps[:, :, 1].sum(1) == [n_phot[i].sum() for i in range(n_sim_per_chunk)]), \
                    "Photons counts in maps and n_phot lists are not consistent! Aborting..."

                # Collect garbage
                auto_garbage_collect()

                # Save settings
                if chunk == 0 and int(job_id) == 0:
                    settings_out = dict()
                    settings_out["T"] = t
                    settings_out["priors"] = prior_dict[temp]
                    settings_out["exp"] = exp  # exposure
                    settings_out["rescale_compressed"] = rescale_compressed
                    settings_out["max_NP_sources"] = np.nan  # not set here
                    settings_out["indices_roi"] = np.argwhere(1 - total_mask_neg).flatten()
                    settings_out["format"] = "NEST"
                    settings_out["mask_type"] = mask_type
                    settings_out["outer_rad"] = outer_rad
                    settings_out["inner_band"] = inner_band
                    settings_out["leakage_delta"] = leakage_delta
                    settings_out["nside"] = nside
                    print("   Writing settings file...")
                    with open(os.path.join(temp_folder, name + "_settings.pickle"), 'wb') as f:
                        pickle.dump(settings_out, f)

                # Save maps
                data_out["data"] = (sim_maps[:, temp_dict["indices_roi"], :]).astype(dtype_data)
                data_out["n_phot"] = n_phot
                data_out["flux_arr"] = [np.asarray(f, dtype=dtype_flux_arr) for f in flux_arr]
                data_out["info"] = dict()
                data_out["info"]["tot_flux"] = np.asarray([np.sum(f) for f in flux_arr])
                data_out["info"]["means"] = mean_draw
                data_out["info"]["vars"] = var_draw
                data_out["info"]["skew"] = skew_draw

                with open(os.path.join(temp_folder, name + "_"
                                                    + str(job_id) + "_" + str(chunk) + ".pickle"), 'wb') as f:
                    pickle.dump(data_out, f)

                # Plot some maps and save
                if chunk == 0 and int(job_id) == 0 and save_example_plot:
                    plt.ioff()
                    hp.mollview(t * (1 - total_mask_neg), title="Template (not exposure-corrected)", nest=True)
                    hp.mollview(exp, title="Exposure (nside = " + str(nside) + ")", nest=True)
                    hp.mollview(total_mask_neg, title="Mask (" + str(mask_type) + ")", nest=True)
                    hp.mollview(total_mask_neg_safety, title="Extended mask (allowing leakage into ROI)", nest=True)
                    for i in range(n_example_plots):
                        hp.mollview(sim_maps[i, :, 0], title=int(np.round(sim_maps[i, :, 0].sum())), nest=True)

                    multipage(os.path.join(output_path, temp + "_examples.pdf"))
                    plt.close("all")

    dash = 80 * "="
    print(dash)
    print("Done! Computation took {0} seconds.".format(time.time() - start_time))
    print(dash)
    # Loading pickle file e.g.: data = pickle.load( open( "./data/<...>.pickle", "rb" ) )
