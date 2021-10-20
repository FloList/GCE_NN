"""
Compare NN and NPTFit predictions for the Fermi map. Train NN first and run NPTFit.
"""
import numpy as np
import healpy as hp
import sys
from NPTFit import create_mask as cm  # Module for creating masks
from gce_utils import get_template, mkdir_p
from matplotlib import pyplot as plt
import os
import pickle
import seaborn as sns
import scipy as sp
from scipy.integrate import trapz
import colorcet as cc
import matplotlib as mpl

sns.set_context("talk")
sns.set_style("ticks")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 14
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
######################################################
if "flo" in os.path.expanduser("~"):
    GADI = False
else:
    GADI = True
if not GADI:
    sys.path.append('/home/flo/PycharmProjects/GCE/MultiNest/lib')
new_lib = '/usr/local/lib'
try:
    if new_lib not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ':'+new_lib
except:
    os.environ['LD_LIBRARY_PATH'] = new_lib
######################################################
EXP = "Fermi_example_p6v11_low_n1"  # see below for the templates used in each experiment

# (NPTFit) Settings
M = True  # apply a PS mask
nside = 128
ROI_ring = 25
nexp = 5  # number of exposure regions
nlive = 500  # number of live points
TASK = EXP + "_M" if M else EXP + '_UNM'
fermi_folder = '/scratch/u95/fl9575/GCE_v2/data/Fermi_Data/fermi_data_573w' if GADI \
    else '/home/flo/PycharmProjects/GCE/data/Fermi/fermi_data_573w'
fermi_folder += "/fermi_data_" + str(nside)

# Templates
all_models_P = ["iso", "dif", "bub", "gce", "gce_12", "dif_O_pibs", "dif_O_ic", "psc"]
all_models_NP = ["iso_PS", "disk_PS", "gce_PS", "gce_12_PS", "bub_PS"]

#############
# CHOOSE MODELS
if EXP in ["Fermi_example", "Fermi_example_low_n1"]:
    T_P = ["dif_O_pibs", "dif_O_ic", "iso", "bub"]
    T_NP = ["gce_12_PS", "thin_disk_PS"]
elif EXP in ["Fermi_example_p6v11", "Fermi_example_p6v11_low_n1"]:
    T_P = ["dif", "iso", "bub"]
    T_NP = ["gce_12_PS", "thin_disk_PS"]
else:
    raise NotImplementedError
#############

# FERMI DATA / MOCK DATA
glob_indices = [0]
counts = np.load(os.path.join(fermi_folder, 'fermidata_counts.npy'))

if len(counts.shape) == 1:
    counts = counts[None]

# Exposure map
exp = np.load(os.path.join(fermi_folder, 'fermidata_exposure.npy'))
mean_exp = np.mean(exp)
cor_term = np.log10(mean_exp)
rescale = exp / mean_exp

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Define priors for parameter estimation
prior_dict = dict()

# NOTE: For large values of A, extremely many sources -> takes very long! BUT: option to set max_NP_sources
# Also, things become weird if n below the smallest break is larger than 1 (lam < 0 error...) -> set < 1

# Poissonian templates: A
prior_dict["iso"] = [-3, 2]
prior_dict["dif"] = [0, 2]
prior_dict["dif_O_pibs"] = [0, 2]
prior_dict["dif_O_ic"] = [0, 2]
prior_dict["bub"] = [-3, 2]
prior_dict["gce"] = prior_dict["gce_12"] = [-3, 2]
prior_dict["psc"] = [-3, 2]

# Non-Poissonian templates: A, n_1, .., n_k, S_1, ... S_{k-1}
# prior_dict["iso_PS"] = [[-6, -1], [2.05, 5], [0.5, 4.5], [-1.95, 1.95], [1, 40], [0.05, 40]]
prior_dict["iso_PS"] = [[-6, 2], [2.05, 100], [-5, 1.95], [0.05, 60]]
prior_dict["gce_PS"] = [[-6, 1], [2.05, 100], [-5, 1.95], [0.05, 60]]
prior_dict["bub_PS"] = prior_dict["gce_12_PS"] = prior_dict["gce_PS"]
prior_dict["disk_PS"] = [[-6, 2], [2.05, 100], [-5, 1.95], [0.05, 60]]
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

# SET TASK_RANGE FOR THIS PROCESS
n_samples = counts.shape[0]  # no. samples

if n_samples == 1:
    TASK_RANGE = [0]
else:
    raise NotImplementedError

for i_task in TASK_RANGE:
    loc_task = TASK + "/sample_" + str(glob_indices[i_task])
    print("Starting with task", loc_task)
    if i_task == TASK_RANGE[0]:
        mkdir_p(os.path.join("/scratch/u95/fl9575/GCE_v2/chains", TASK) if GADI
                else os.path.join("/home/flo/PycharmProjects/GCE/chains", TASK))
    loc_folder = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", loc_task) if GADI \
        else os.path.join("/home/flo/PycharmProjects/GCE/chains", loc_task)
    mkdir_p(loc_folder)

    n = nptfit.NPTF(tag=loc_task)
    exp_for_fitting = exp
    n.load_data(counts[i_task, :].astype(np.int32), exp_for_fitting)

    # Add mask
    total_mask_neg = cm.make_mask_total(nside=nside, band_mask=True, band_mask_range=2, mask_ring=True, inner=0, outer=ROI_ring)
    if M:
        pscmask = np.load(os.path.join(fermi_folder, "fermidata_pscmask_3fgl.npy"))
        total_mask_neg = (1 - (1 - total_mask_neg) * (1 - pscmask))

    n.load_mask(total_mask_neg)

    # Add templates for Poissonian models (in terms of counts)
    for temp in T_P:
        T = get_template(fermi_folder, temp)
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
        par_names[0] = '$A_{\mathrm{%s}}^{\mathrm{PS}}$' % temp[:-3]
        par_names[1:len(is_log)//2+1] = ['$n^{\mathrm{%s}}_{\mathrm{%i}}$' % (temp[:-3].replace("_", "-"), i) for i in range(1, len(is_log)//2+1)]
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

    # Analyse
    save_loc = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", loc_task) if GADI else os.path.join(
        "/home/flo/PycharmProjects/GCE/DeepSphere/chains", loc_task)
    # Analyse!
    n.load_scan()
    an = dnds_analysis.Analysis(n)
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
    colours_P = ['#ff0000', '#ec7014', '#fec44f', '#37c837', '#41b6c4']
    colours_NP = ['deepskyblue', 'black']
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

    # Get FFs
    samples_dict = dict()
    loc_dict, flux_dict, plus_dict, minus_dict, plus_dict_FF, minus_dict_FF = dict(), dict(), dict(), dict(), dict(), dict()
    for temp in T_P + T_NP:
        if temp not in flux_dict.keys():
            flux_dict[temp], plus_dict[temp], minus_dict[temp], plus_dict_FF[temp], minus_dict_FF[temp] = [], [], [], [], []

        if temp in T_P:
            samples_loc = an.return_intensity_arrays_poiss(temp.replace("_", "-"))
        else:
            samples_loc = an.return_intensity_arrays_non_poiss(temp.replace("_", "-"), smin=0.01, smax=1000, nsteps=1000)
        samples_dict[temp] = samples_loc

        loc_dict[temp] = np.median(samples_loc)
        plus_dict[temp] = np.quantile(samples_loc, 0.84) - loc_dict[temp]
        minus_dict[temp] = loc_dict[temp] - np.quantile(samples_loc, 0.16)

    total_flux = np.sum([loc_dict[temp] for temp in loc_dict.keys()])

    for temp in T_P + T_NP:
        flux_dict[temp].append(loc_dict[temp] / total_flux)

    for temp in T_P + T_NP:
        plus_dict_FF[temp].append(plus_dict[temp] / total_flux)
        minus_dict_FF[temp].append(minus_dict[temp] / total_flux)

    save_loc = os.path.join("/scratch/u95/fl9575/GCE_v2/chains", TASK) if GADI else os.path.join(
        "/home/flo/PycharmProjects/GCE/DeepSphere/chains", TASK)
    with open(os.path.join(save_loc, "NPTFit_flux_fractions.pickle"), 'wb') as f:
        pickle.dump(flux_dict, f)
        print("Flux dict file written.")

    # Results for Fermi_example_low_n1_M:
    # NPTFit: 52.6+0.6-0.6%, 26.1+1.4-1.3%, 1.8+1.0-1.1%, 5.9+0.5-0.4%, 7.9+-0.4%, 5.7+-1.1%
    # NN:     52.6+-0.7%,    27.0+-1.3%,    2.0+-0.9%,    6.2+-0.5%,    7.9+-0.5%,    4.3+-1.2%
    # (see plot_flux_fraction_posteriors_fermi.py for NN results, produced with evaluate_NN_Fermi.py)
    loc_NN = np.asarray([0.5256651, 0.270391, 0.01978799, 0.06156372, 0.07928304, 0.04330921])  # means:
    unc_NN = np.asarray([0.00744413, 0.01342631, 0.00916832, 0.00452482, 0.00478658, 0.01155945])  # pred. stds

    # Make plot
    std_fac = 4
    lw_NN = 0
    lw_NPTF = 2
    do_fill = True
    colours = ['#ff0000', '#ec7014', '#fec44f', '#37c837', 'deepskyblue', 'k']
    alpha_NN = 0.3

    # Plot NN first
    fig_FFs, ax_FFs = plt.subplots(figsize=(12, 6))
    for i_model in range(len(T_P + T_NP)):
        y_vec = np.linspace(loc_NN[i_model] - std_fac * unc_NN[i_model], loc_NN[i_model] + std_fac * unc_NN[i_model], 1000)
        if do_fill:
            ax_FFs.fill_between(100 * y_vec, sp.stats.norm.pdf(y_vec, loc_NN[i_model], unc_NN[i_model]),
                                color=colours[i_model],
                                lw=lw_NN, linestyle="-", alpha=alpha_NN)
            ax_FFs.plot(100 * y_vec, sp.stats.norm.pdf(y_vec, loc_NN[i_model], unc_NN[i_model]), color=colours[i_model],
                        lw=lw_NN, linestyle="-", label=str(i_model))
        else:
            ax_FFs.plot(100 * y_vec, sp.stats.norm.pdf(y_vec, loc_NN[i_model], unc_NN[i_model]), color=colours[i_model],
                        lw=lw_NN, linestyle="-", label=str(i_model))

    # Now: NPTFit
    n_bins = 250
    for i_model, model in enumerate(T_P + T_NP):
        frac_hist_comp, bin_edges_comp = np.histogram(100*np.array(samples_dict[model]) / total_flux, bins=n_bins,
                                                      range=(0, 100), density=True)
        bin_centres = (bin_edges_comp[1:] + bin_edges_comp[:-1]) / 2
        ax_FFs.plot(bin_centres, 100 * frac_hist_comp, color=colours[i_model], lw=lw_NPTF, linestyle="-", label=str(i_model))

    # Adjust plot
    ax_FFs.set_xlim([0, 57])
    ax_FFs.set_ylim([0, 100])
    xticks = np.arange(0, 60, 10)
    ax_FFs.set_xticks(xticks)
    ax_FFs.set_xlabel(r"Flux fractions [$\%$]")
    ax_FFs.set_ylabel("Probability density")
    plt.tight_layout()
    fig_FFs.savefig(save_loc + "_flux_fraction_comparison.pdf", bbox_inches="tight")

    # dNdF plot
    # plt.figure(figsize=[6, 5])
    # spow = 1  # F dN/dF
    # an.plot_source_count_median('thin-disk-PS', smin=0.01, smax=1000, nsteps=1000, color='royalblue', spow=spow,
    #                             label='Disk PS')
    # an.plot_source_count_band('thin-disk-PS', smin=0.01, smax=1000, nsteps=1000, qs=[0.16, 0.5, 0.84], color='royalblue',
    #                           alpha=0.15, spow=spow)
    # an.plot_source_count_band('thin-disk-PS', smin=0.01, smax=1000, nsteps=1000, qs=[0.025, 0.5, 0.975],
    #                           color='royalblue', alpha=0.1, spow=spow)
    # an.plot_source_count_median('gce-12-PS', smin=0.01, smax=1000, nsteps=1000, color='firebrick', spow=spow,
    #                             label='GCE PS')
    # an.plot_source_count_band('gce-12-PS', smin=0.01, smax=1000, nsteps=1000, qs=[0.16, 0.5, 0.84], color='firebrick',
    #                           alpha=0.15, spow=spow)
    # an.plot_source_count_band('gce-12-PS', smin=0.01, smax=1000, nsteps=1000, qs=[0.025, 0.5, 0.975],
    #                           color='firebrick', alpha=0.1, spow=spow)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlim([1e-13, 1e-8])
    # plt.ylim([1e-10, 1e-7])

    # Compare relative F^2 dNdF
    # NOTE: NN predicts F dN/d(logF) = F^2 dN/dF because bins are log-spaced!
    qs = [0.05, 0.5, 0.95]
    an.calculate_dndf_arrays("gce-12-PS", smin=0.01, smax=1000, nsteps=1000, qs=qs)
    farray = an.flux_array.copy()
    gce_12_PS_dNdF_median = an.qmid.copy()
    gce_12_PS_F2_dNdF_median = gce_12_PS_dNdF_median * farray ** 2
    gce_12_PS_F2_dNdF_median_norm = gce_12_PS_F2_dNdF_median / gce_12_PS_F2_dNdF_median.sum()
    an.calculate_dndf_arrays("thin-disk-PS", smin=0.01, smax=1000, nsteps=1000, qs=qs)
    thin_disk_dNdF_median = an.qmid.copy()
    thin_disk_F2_dNdF_median = thin_disk_dNdF_median * farray ** 2
    thin_disk_F2_dNdF_median_norm = thin_disk_F2_dNdF_median / thin_disk_F2_dNdF_median.sum()
    NPTF_pred = np.vstack([gce_12_PS_F2_dNdF_median_norm, thin_disk_F2_dNdF_median_norm]).T
    NPTF_pred_abs = np.vstack([gce_12_PS_F2_dNdF_median, thin_disk_F2_dNdF_median]).T

    NN_pred_data = np.load("./checkpoints/Fermi_example_add_two_256_BN_bs_256_softplus_pre_gen/fermi_prediction.npz",
                           allow_pickle=True)
    NN_pred = NN_pred_data["fermi_pred"][()]["gce_hist"]
    tau_vec = NN_pred_data["tau_vec"][()]
    bin_centres = NN_pred_data["bin_centres"][()]

    make_plot_fermi_NN_and_NPTFit(NN_pred, NPTF_pred, np.log10(farray), tau_vec, bin_centres, name="fermi_comparison.pdf",
                                  mean_exp=mean_exp)

    # Also make a plot of the absolute F^2 dN/dF
    tot_fermi_flux_ROI_nside_256 = 1.7465961267807914e-06
    tot_fermi_flux_ROI_NPTFit = an.total_counts / an.exp_masked_mean
    template_inds_NN = [4, 5]
    NN_pred_FFs = NN_pred_data["fermi_pred"][()]["logits_mean"].mean(0)  # avg. over taus doesn't do anything
    F_tot_templates = tot_fermi_flux_ROI_nside_256 * NN_pred_FFs[template_inds_NN]
    F_tot_templates_NPTF = tot_fermi_flux_ROI_NPTFit * np.asarray([flux_dict['gce_12_PS'], flux_dict['thin_disk_PS']]).flatten()

    norm_fac = np.asarray([[trapz(NN_pred[i_tau, :, i_temp] / (10 ** bin_centres), (10 ** bin_centres)) / F_tot_templates[i_temp]
                            for i_tau in range(len(tau_vec))] for i_temp in range(2)]).T
    F2dNdF_abs = NN_pred / np.expand_dims(norm_fac, 1)

    norm_fac_NPTF = np.asarray([trapz(NPTF_pred_abs[:, i_temp] / farray, farray) / F_tot_templates_NPTF[i_temp] for i_temp in range(2)])
    NPTF_pred_abs_rescaled = NPTF_pred_abs / norm_fac_NPTF[None]    # rescaled to the same units as NN
    # number of PSs: trapz(NPTF_pred_abs_rescaled[:, 0] / farray ** 2, farray)

    # Plot NN and NPTFit
    fig_abs, axs_abs = plt.subplots(1, 2, figsize=(8, 4), sharex="all")
    colors = cc.cm.bkr(np.linspace(0, 1, len(tau_vec)))[::-1]
    one_ph_flux = np.log10(1 / an.exp_masked_mean)

    for i_temp in range(2):
        # Plot NN
        for i_tau, tau in enumerate(tau_vec):
            axs_abs[i_temp].plot(bin_centres, F2dNdF_abs[i_tau, :, i_temp], color=colors[i_tau], lw=2)

        # Plot NPTF median
        axs_abs[i_temp].plot(np.log10(farray), NPTF_pred_abs_rescaled[:, i_temp], color="#00d4aa", lw=2)

        axs_abs[i_temp].set_xlabel(r"$\log_{10} \ F$")
        axs_abs[i_temp].axvline(one_ph_flux, color="orange", ls="--", zorder=4)
        rect = mpl.patches.Rectangle((np.log10(4e-10), 0.0), np.log10(5e-10) - np.log10(4e-10), axs_abs[i_temp].get_ylim()[-1],
                                     linewidth=0, edgecolor=None, facecolor="#ccccccff", zorder=-1)
        axs_abs[i_temp].add_patch(rect)
    axs_abs[0].set_ylabel(r"$F^2 \, \frac{dN}{dF}$")
    plt.tight_layout()
