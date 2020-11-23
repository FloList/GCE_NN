"""
This script performs an NPTFit for the Fermi counts or for mock data (supports multiple maps).
see https://github.com/bsafdi/NPTFit/tree/master/examples
"""
import numpy as np
import healpy as hp
import sys
from NPTFit import create_mask as cm  # Module for creating masks
from gce_utils import get_template, mkdir_p, masked_to_full, auto_garbage_collect
from matplotlib import pyplot as plt
from shutil import copyfile
import time
import os
import pickle
import seaborn as sns
import timeout_decorator
from mpi4py import MPI
sns.set_context("talk")
sns.set_style("white")
######################################################
if "flo" in os.path.expanduser("~"):
    HPC = False
else:
    HPC = True
######################################################
FIT = True   # Fit or load?
EXP = "GCE_for_letter_p6v11"  # see below for the templates used in each experiment
RESUME = False  # resume fitting?
MOCK_DATA = False  # use (best-fit) mock data instead of real Fermi data
CONST_exp = False  # Map is assumed to be generated using a constant exposure map, with mean Fermi exposure
# # # # # # # # # # # # # # # # # # # # # #

# Settings
M = False  # apply a mask
nside = 128
ROI_ring = 25
nexp = 1 if CONST_exp else 5  # number of exposure regions
nlive = 500  # number of live points
max_time = 2 * 60 * 60  # max. time for sampling in seconds (per sample)
TASK = EXP + "_M" if M else EXP + '_UNM'
if MOCK_DATA:
    TASK += "_m"
if CONST_exp:
    TASK += "_c"
fermi_folder = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data' if HPC else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data'

# Settings for mock data
# mock_folder = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_mock' if HPC else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_mock'
mock_folder = '/scratch/u95/fl9575/GCE_v2/data/GCE_and_background_12_const_exp/' if HPC else '/home/flo/Documents/Latex/GCE/Data_for_paper/GCE_and_background_12_const_exp/'
mock_file = "GCE_and_background_12_const_exp_subset_256.pickle"
settings_file = "GCE_maps_settings.pickle"
mock_is_npy = False  # if True: mock data in a single .npy file, if False: mock data as pickle files

# Templates
all_models_P = ["iso", "dif", "bub", "gce", "gce_12", "dif_O_pibs", "dif_O_ic", "psc"]
all_models_NP = ["iso_PS", "disk_PS", "gce_PS", "gce_12_PS", "bub_PS"]

#############
# CHOOSE MODELS
if EXP is "dif_O":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
    T_NP = ["iso_PS", "disk_PS", "gce_12_PS"]
elif EXP is "dif_O_psc":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic", "psc"]
    T_NP = ["gce_12_PS"]
elif EXP is "dif_O_no_iso_PS":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
    T_NP = ["disk_PS", "gce_12_PS"]
elif EXP is "LS":
    T_P = ["iso", "bub", "gce_12", "dif"]
    T_NP = ["iso_PS", "disk_PS", "gce_12_PS"]
elif EXP is "GCE_toy":
    T_P = ["dif", "gce_12"]
    T_NP = ["gce_12_PS"]
elif EXP is "GCE_and_background":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
    T_NP = ["gce_12_PS"]
elif EXP is "GCE_for_letter":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
    T_NP = ["gce_12_PS", "thin_disk_PS"]
elif EXP is "GCE_for_letter_thick":
    T_P = ["iso", "bub", "gce_12", "dif_O_pibs", "dif_O_ic"]
    T_NP = ["gce_12_PS", "thick_disk_PS"]
elif EXP is "GCE_for_letter_p6v11":
    T_P = ["iso", "bub", "gce_12", "dif"]
    T_NP = ["gce_12_PS", "thin_disk_PS"]
else:
    raise NotImplementedError
#############

# FERMI DATA / MOCK DATA
glob_indices = [0]
if MOCK_DATA:
    if mock_is_npy:
        counts = np.load(os.path.join(mock_folder, EXP + ".npy"))
    else:
        all_files = os.listdir(mock_folder)

        # Setting file should be stored in the mock folder
        settings_ind = np.argwhere([settings_file == file for file in all_files])[0][0]
        settings_file = open(os.path.join(mock_folder, all_files[settings_ind]), 'rb')
        settings_dict = pickle.load(settings_file)
        settings_file.close()
        data_ind = np.argwhere([mock_file == file for file in all_files])[0][0]
        data_file = open(os.path.join(mock_folder, all_files[data_ind]), 'rb')
        data_dict = pickle.load(data_file)
        data_file.close()
        counts = masked_to_full(data_dict["data"].T, settings_dict["unmasked_pix"], nside=nside)

        # first and second input arguments can be used to specify the range of samples that shall be used
        counts_start = 0 if len(sys.argv) < 2 else int(sys.argv[1])
        counts_end = counts.shape[0] if len(sys.argv) < 3 else int(sys.argv[2]) + 1
        glob_indices = range(counts_start, counts_end)

        counts = counts[counts_start:counts_end, :]
else:
    counts = np.load(os.path.join(fermi_folder, 'fermidata_counts.npy'))

if len(counts.shape) == 1:
    counts = counts[None]

# Exposure map
exp = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
mean_exp = np.mean(exp)
cor_term = np.log10(mean_exp)
rescale = exp / mean_exp

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Define priors for parameter estimation (ROUGHLY: see Leane Table S1)
prior_dict = dict()

# NOTE: For large values of A, extremely many sources -> takes very long!
# Poissonian templates: A
prior_dict["iso"] = [-3, 2]  # originally: upper limit 1
prior_dict["dif"] = [0, 2]
prior_dict["dif_O_pibs"] = [0, 2]
prior_dict["dif_O_ic"] = [0, 2]
prior_dict["bub"] = [-3, 2]  # originally: upper limit 1
prior_dict["gce"] = prior_dict["gce_12"] = [-3, 2]  # originally: upper limit 1
prior_dict["psc"] = [-3, 2]

# Non-Poissonian templates: A, n_1, .., n_k, S_1, ... S_{k-1}
prior_dict["iso_PS"] = [[-6, -1], [2.05, 5], [0.5, 4.5], [-1.95, 0.95], [1, 40], [0.05, 40]]
prior_dict["gce_PS"] = [[-6, 1], [5, 60], [-3, 0.95], [0.05, 60]]
prior_dict["bub_PS"] = prior_dict["gce_12_PS"] = prior_dict["gce_PS"]
prior_dict["disk_PS"] = [[-6, 2], [2.05, 60], [-3, 0.95], [0.05, 60]]
prior_dict["thin_disk_PS"] = prior_dict["thick_disk_PS"] = prior_dict["disk_PS"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Fit using NPTFit
from NPTFit import nptfit
from NPTFit import psf_correction as pc
from NPTFit import dnds_analysis

if not FIT:
    flux_dict = dict()

# SET TASK_RANGE FOR THIS PROCESS
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_cpus = comm.Get_size()  # no. cpus
n_samples = counts.shape[0]  # no. samples

if n_samples is 1:
    TASK_RANGE = [0]
else:
    n_cpus_per_sample = 4  # no. cpus per samples
    n_blocks = int(np.ceil(n_cpus / n_cpus_per_sample))  # number of blocks
    i_block = int(np.ceil((rank + 1) / n_cpus_per_sample)) - 1  # block of the current cpu
    ds = n_samples // n_blocks
    start_samples = np.floor(np.linspace(0, n_samples - ds, n_blocks)).astype(int)
    TASK_RANGE = range(start_samples[-1], n_samples) if i_block == n_blocks - 1 else range(start_samples[i_block], start_samples[i_block+1])
    print("This is process", rank+1, "out of", n_cpus, "in computation block", i_block+1, "out of", n_blocks, "taking care of samples '", TASK_RANGE, "'")

for i_task in TASK_RANGE:
    loc_task = TASK + "/sample_" + str(glob_indices[i_task])
    print("Starting with task", loc_task)
    if i_task == TASK_RANGE[0]:
        mkdir_p(os.path.join("/scratch/u95/fl9575/GCE_v2/chains", TASK) if HPC else os.path.join("/home/flo/PycharmProjects/GCE/chains", TASK))
    loc_folder = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", loc_task) if HPC else os.path.join("/home/flo/PycharmProjects/GCE/chains", loc_task)
    mkdir_p(loc_folder)

    n = nptfit.NPTF(tag=loc_task)
    exp_for_fitting = np.ones_like(exp) * mean_exp if CONST_exp else exp
    n.load_data(counts[i_task, :].astype(np.int32), exp_for_fitting)

    # Add mask
    total_mask_neg = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=ROI_ring)
    if M:
        pscmask = np.load(os.path.join(fermi_folder, "fermidata_pscmask.npy"))
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - pscmask))

    n.load_mask(total_mask_neg)

    # Add templates for Poissonian models (in terms of counts)
    for temp in T_P:
        T = get_template(fermi_folder, temp)
        if CONST_exp:
            # need to remove the template normalisation!
            T /= rescale
        n.add_template(T, temp.replace("_", "-"))

    # Add templates for Non-Poissonian models (exposure correction removed!)
    for temp in T_NP:
        T = get_template(fermi_folder, temp[:-3])
        # always need to remove the template normalisation!
        T_corr = T / rescale
        n.add_template(T_corr, temp.replace("_", "-"), units='PS')

    # Add Poissonian models
    for temp in T_P:
        n.add_poiss_model(temp.replace("_", "-"), '$A_{\mathrm{%s}}$'%temp.replace("_", "-"), prior_dict[temp], True)  # A is log quantity!

    # Add the Non-Poissonian models
    for temp in T_NP:
        # Set is_log list
        is_log = False * np.ones(len(prior_dict[temp])).astype(bool)
        is_log[0] = True
        # Set parameter names
        par_names = [None] * len(is_log)
        par_names[0] = '$A_{\mathrm{%s}}^{\mathrm{PS}}$'%temp[:-3]
        par_names[1:len(is_log)//2+1] = ['$n^{\mathrm{%s}}_{\mathrm{%i}}$'%(temp[:-3].replace("_", "-"), i) for i in range(1, len(is_log)//2+1)]
        par_names[len(is_log)//2+1:] = ['$S^{\mathrm{%s}}_{\mathrm{%i}}$' % (temp[:-3].replace("_", "-"), i) for i in range(1, len(is_log)//2)]
        # Add model
        n.add_non_poiss_model(temp.replace("_", "-"), par_names, prior_dict[temp], is_log)

    pc_inst = pc.PSFCorrection(delay_compute=True)
    pc_inst.psf_r_func = lambda r: Fermi_PSF(r)
    pc_inst.sample_psf_max = 10.*spe*(score+stail)/2.
    pc_inst.psf_samples = 10000
    pc_inst.psf_tag = 'Fermi_PSF_2GeV'
    pc_inst.make_or_load_psf_corr()

    f_ary = pc_inst.f_ary
    df_rho_div_f_ary = pc_inst.df_rho_div_f_ary

    n.configure_for_scan(f_ary, df_rho_div_f_ary, nexp=nexp)

    if FIT:
        # Back up this file
        if i_task == TASK_RANGE[0]:
            datetime = time.ctime().replace("  ", "_").replace(" ", "_").replace(":", "-")
            save_loc = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", TASK) if HPC else os.path.join("/home/flo/PycharmProjects/GCE/DeepSphere/chains", TASK)
            mkdir_p(save_loc)
            main_backup = os.path.join(save_loc, os.path.split(__file__)[-1][:-3] + "_" + datetime + ".py")
            copyfile(__file__, main_backup)

        @timeout_decorator.timeout(max_time, timeout_exception=StopIteration, use_signals=False)
        def do_scan(n_, nlive_, pymultinest_options_=None):
            if pymultinest_options_ is not None:
                n_.perform_scan(nlive=nlive_, pymultinest_options=pymultinest_options_)
            else:
                n_.perform_scan(nlive=nlive_)

        pymultinest_options = {'importance_nested_sampling': False,
                                   'resume': True, 'verbose': True,
                                   'sampling_efficiency': 'model',
                                   'init_MPI': False, 'evidence_tolerance': 0.5,
                                   'const_efficiency_mode': False} if RESUME else None
        try:
            do_scan(n, nlive_=nlive, pymultinest_options_=pymultinest_options)
        except StopIteration:
            save_loc = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", loc_task) if HPC else os.path.join(
                "/home/flo/PycharmProjects/GCE/DeepSphere/chains", loc_task)
            print("Process", rank+1, "ran out of time while sampling (max_time = {}s)! Continuing with the next sample...".format(max_time))
            np.save(os.path.join(save_loc, str(rank+1) + "_not_finished"), [True])

        auto_garbage_collect()

    else:
        save_loc = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", loc_task) if HPC else os.path.join(
            "/home/flo/PycharmProjects/GCE/DeepSphere/chains", loc_task)
        # Analyse!
        try:
            n.load_scan()
        except IndexError:
            for temp in T_P:
                if temp not in flux_dict.keys():
                    flux_dict[temp] = []
            for temp in T_NP:
                if temp not in flux_dict.keys():
                    flux_dict[temp] = []
            for temp in T_P + T_NP:
                flux_dict[temp].append(np.nan)
                print("Fit for this sample didn't finish!")
            continue

        an = dnds_analysis.Analysis(n)
        if not HPC:
            plt.ion()
        labels = [i.replace("\\", "").replace("mathrm", "").replace("'", "").replace("-", "") for i in list(n.params)]
        n_params = len(n.params)

        # Triangle plot
        an.make_triangle()
        fig = plt.gcf()
        fig.savefig(save_loc + "_corner_plot.pdf", bbox_inches="tight")

        # Best fit parameters
        best_params = an.get_best_fit_params()
        fig_0, ax_0 = plt.subplots(1, 1, figsize=(16, 16))
        ax_0.bar(range(n_params), best_params)
        ax_0.set_xticks(range(n_params))
        ax_0.set_xticklabels(labels, size=14)
        ax_0.set_title("Best fit parameters")
        for i, v in enumerate(an.get_best_fit_params()):
            ax_0.text(i, (v+1) if v > 0 else (v-1) , str(np.round(v, 2)), fontweight='bold', ha='center', va='center')
        fig_0.savefig(save_loc + "_best_fit.pdf", bbox_inches="tight")

        # Plot flux fractions
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(16, 16))
        colours_P = ['#ff0000', '#ec7014', '#fec44f', '#37c837', '#41b6c4', '#225ea8', '#000000']
        colours_NP = ['deepskyblue', 'darkslateblue', 'black', 'olivedrab', 'seagreen',  'teal']
        nbins = 1500
        for i_temp, temp in enumerate(T_P):
            an.plot_intensity_fraction_poiss(temp.replace("_", "-"), bins=nbins, color=colours_P[i_temp], label=temp, lw=3)
        for i_temp, temp in enumerate(T_NP):
            an.plot_intensity_fraction_non_poiss(temp.replace("_", "-"), bins=nbins, color=colours_NP[i_temp], label=temp, lw=3)
        ax_1.set_xlabel('Flux fraction (%)')
        ax_1.legend(fancybox=True)
        ax_1.set_xlim(0, 60)
        ax_1.set_ylim(0, .1)
        fig_1.savefig(save_loc + "_flux_fractions.pdf", bbox_inches="tight")

        loc_dict = dict()
        for temp in T_P:
            if temp not in flux_dict.keys():
                flux_dict[temp] = []
            loc_dict[temp] = np.median(an.return_intensity_arrays_poiss(temp.replace("_", "-")))
        for temp in T_NP:
            if temp not in flux_dict.keys():
                flux_dict[temp] = []
            loc_dict[temp] = np.median(an.return_intensity_arrays_non_poiss(temp.replace("_", "-")))

        total_flux = np.sum([loc_dict[temp] for temp in loc_dict.keys()])

        for temp in T_P + T_NP:
            flux_dict[temp].append(loc_dict[temp] / total_flux)

        plt.close("all")
if not FIT:
    save_loc = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", TASK) if HPC else os.path.join(
        "/home/flo/PycharmProjects/GCE/DeepSphere/chains", TASK)
    with open(os.path.join(save_loc, "NPTFit_flux_fractions.pickle"), 'wb') as f:
        pickle.dump(flux_dict, f)
        print("Flux dict file written.")


# Helper function (adapted from NPTFit)
def residual_map_for_theta(an, theta, mask=False, maskval=0., exclude=np.array([]),
                 smooth=False, smooth_sig=1.):
    """ Return the residual map, which is the data minus the best fit
        Poissonian templates
        :param an: analysis
        :param theta: parameter vector
        :param mask: whether to apply a mask to the residual map
        :param maskval: value masked pixels are set to if mask=True
        :param exclude: array of strings, listing which templates not to
        subtract
        :param smooth: whether to smooth the map to scale smooth_sig
        :param smooth_sig: std dev of the smoothing kernel [degrees]
    """

    residual = np.array(an.nptf.count_map).astype(np.float)

    # Loop through floated and fixed models
    fix_keys = an.nptf.poiss_models_fixed.keys()
    for k in fix_keys:
        # Skip excluded maps
        if k in exclude: continue

        tmp = an.nptf.templates_dict[k]
        norm = an.nptf.poiss_models_fixed[k]['fixed_norm']

        residual -= norm * tmp

    if an.nptf.n_poiss != 0:
        flt_keys = an.nptf.poiss_model_keys
        theta_bf = theta

        # Get the best fit values, convert from log if need be
        a_theta = np.array([an.nptf.convert_log(
            an.nptf.model_decompression_key[i], theta_bf[i])[1]
                            for i in range(an.nptf.n_poiss)])

        mdk = np.array(an.nptf.model_decompression_key)[:, 0]

        for k in flt_keys:
            # Skip excluded maps
            if k in exclude: continue

            tmp = an.nptf.templates_dict[k]

            model_where = np.where(mdk == k)[0][0]
            norm = a_theta[model_where]

            residual -= norm * tmp

    # Smooth
    if smooth:
        # Convert smoothing scale to radians
        sigma_psf = smooth_sig * np.pi / 180.
        residual = hp.smoothing(residual, sigma=sigma_psf)

    # Mask
    if mask:
        msk = an.nptf.mask_total

        tomask = np.where(msk == True)[0]

        residual[tomask] = maskval

    return residual
