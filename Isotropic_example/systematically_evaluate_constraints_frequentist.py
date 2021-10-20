"""
Get constraints of the Poisson fraction for the maps generated with
   'systematically_evaluate_constraints_part_1.py'
from frequentist LLH-based approach (see file 'Poisson95Lim.py').
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import time
from gce_utils import *
from Poisson95Lim import lim95
from ray_progress_bar import ProgressBar
import os
import copy
import ray
import seaborn as sns
plt.ion()

# SET CHI^2 FOR EVALUATION
CHI2_EVAL = 2.71


@ray.remote
def get_single_constraint(m, pba):
    pba.update.remote(1)
    return lim95(m.astype(int), CHI2_EVAL)


# ######################################################################################################################
if __name__ == '__main__':

    # Plot settings
    sns.set_context("talk")
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 14
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rcParams['image.cmap'] = 'rocket'

    # Define paths
    DO_FAINT = True
    if DO_FAINT:
        checkpoint_path = '/scratch/u95/fl9575/GCE_v2/checkpoints/' \
                          'Iso_maps_combined_add_two_faint_no_PSF_IN_bs_256_softplus_pre_gen'
    else:
        checkpoint_path = '/scratch/u95/fl9575/GCE_v2/checkpoints/' \
                          'Iso_maps_combined_add_two_no_PSF_IN_bs_256_softplus_pre_gen'
    sys_data_path = os.path.join(checkpoint_path, "Mixed_PS_Poisson", "Systematic")
    sys_data_file = os.path.join(sys_data_path, "Delta_dNdF_data.npy")
    sys_data_freq_constraints_file = os.path.join(sys_data_path, "constraints_frequentist.npy")

    # Load maps: shape: Poisson fraction x counts per PS x realisations x pixels x 2 (Poiss, PS)
    sys_maps = np.load(sys_data_file, allow_pickle=True)
    n_Poiss_frac, n_counts_per_PS, n_realisations, n_pix_in_ROI, _ = sys_maps.shape
    tot_counts = sys_maps.sum(3)

    # Get final maps by summing Poiss + PS
    total_maps = sys_maps.sum(4)
    max_exp_counts_per_PS_ind = n_counts_per_PS  # take all the counts per PS
    n_realisations_max = n_realisations  # take all the realisations
    final_maps = total_maps[:, :max_exp_counts_per_PS_ind, :n_realisations_max, :]

    # if frequentist constraints have not been computed yet: compute them now
    if not os.path.exists(sys_data_freq_constraints_file):

        # Flatten maps over the parameter grid (Poisson frac x counts per PS x realisations)
        final_maps_flat = np.reshape(final_maps, [-1, final_maps.shape[3]])
        n_eval = final_maps_flat.shape[0]
        # num_cpus = psutil.cpu_count(logical=False)
        num_cpus = 16
        print("Running Ray on", num_cpus, "CPUs.")

        def run():
            ray.init(num_cpus=num_cpus, memory=2000000000, object_store_memory=2000000000)  # memory, object_store_memory can be set
            pb = ProgressBar(n_eval)
            actor = pb.actor
            tasks_pre_launch = [get_single_constraint.remote(final_maps_flat[i], actor) for i in range(n_eval)]

            pb.print_until_done()
            tasks = np.asarray(ray.get(tasks_pre_launch))

            tasks == list(range(n_eval))
            n_eval == ray.get(actor.get_counter.remote())

            return tasks

        # Run
        constraints_95_flat = run()

        # Reshape: n_Poiss_frac x n_counts_per_PS x n_realisations
        constraints_95 = np.reshape(constraints_95_flat, [n_Poiss_frac, max_exp_counts_per_PS_ind, n_realisations_max])

        # Save the constraints
        np.save(sys_data_freq_constraints_file, constraints_95)

    # Load the constraints
    else:
        constraints_95 = np.load(sys_data_freq_constraints_file, allow_pickle=True)
        n_Poiss_frac, max_exp_counts_per_PS_ind, n_realisations_max = constraints_95.shape

    # Define counts per PS and Poisson FF arrays
    counts_per_PS_ary = np.logspace(-1, 3, 11)[:max_exp_counts_per_PS_ind]
    Poiss_fracs = np.linspace(0.0, 1.0, 6)

    # Define the colour map: same limits as CET_D3_r, but going through dark grey (as diverging_gkr_60_10_c40_r)
    x_vec = np.asarray([0, 0.5, 1])
    cmap_orig = copy.copy(cc.cm.CET_D3_r)
    cmap_3_vals = cmap_orig(x_vec)

    cmap_gkr_orig = copy.copy(cc.cm.diverging_gkr_60_10_c40_r)
    cmap_gkr_3_vals = cmap_gkr_orig(x_vec)
    cmap_new_3_vals = np.vstack([cmap_3_vals[0], cmap_gkr_3_vals[1], cmap_3_vals[2]])
    N_interp = 256
    cmap_new_ary = np.asarray([np.interp(np.linspace(0, 1, N_interp), x_vec, cmap_new_3_vals[:, i]) for i in range(4)]).T
    cmap_new = mpl.colors.ListedColormap(cmap_new_ary)
    colors_constraint = cmap_new(np.linspace(0, 1, n_Poiss_frac))

    # Make a plot
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.2))
    ax.set_xscale("log")
    x_values = counts_per_PS_ary
    alpha = 0.18
    lw = 1.5
    for i_Poiss_frac, Poiss_frac in enumerate(Poiss_fracs):
        median_constraint = np.median(constraints_95[i_Poiss_frac], 1)  # median over the maps
        scatter_low = median_constraint - np.quantile(constraints_95[i_Poiss_frac], 0.16, 1)
        scatter_high = np.quantile(constraints_95[i_Poiss_frac], 0.84, 1) - median_constraint
        yerr = np.vstack([scatter_low, scatter_high])
        ax.errorbar(x=x_values, y=median_constraint, yerr=yerr, lw=lw, color=colors_constraint[i_Poiss_frac], capsize=3,
                    marker="o", ms=4, markeredgewidth=1, elinewidth=lw, zorder=2, alpha=alpha)
        #ax.axhline(Poiss_frac, color=colors_constraint[i_Poiss_frac], ls="--", lw=1, zorder=1)
        # if i_Poiss_frac == 0:
            # ax.text(0.1, Poiss_frac + 0.025, r"True $\eta_P$", color=colors_constraint[i_Poiss_frac], size="small")
    ax.set_xlabel("Expected counts per PS")
    ax.set_ylabel(r"Poisson flux fraction $\eta_P$")
    plt.tight_layout()
    fig.savefig(os.path.join(sys_data_path, "systematic_constraints_frequentist_small.pdf"), bbox_inches="tight")

    # Investigate the constraints in detail
    ind_Poiss_frac = -1
    ind_counts_per_PS = -1
    sys_maps_sel = sys_maps[ind_Poiss_frac, ind_counts_per_PS, :, :, :]
    tot_counts_sel_DM_PS = sys_maps_sel.sum(1)
    print("Mean total counts: DM:", tot_counts_sel_DM_PS.mean(0)[0], "PS:", tot_counts_sel_DM_PS.mean(0)[1])
    constraints_95_sel = constraints_95[ind_Poiss_frac, ind_counts_per_PS]
    print("Constraint for each map:\n", np.round(100 * constraints_95_sel, 1))
    print("Total counts for each map:\n", tot_counts_sel_DM_PS.sum(1))

    # Plot some maps
    def zoom_ax(ax):
        ax.set_xlim([-27, 27])
        ax.set_ylim([-27, 27])


    plot_inds = np.arange(-5, 0)
    unmasked_pix = np.load(os.path.join(sys_data_path, "unmasked_pix.npy"))
    do_plot = False
    for i_plot in plot_inds:
        if do_plot:
            hp.cartview(masked_to_full(sys_maps_sel[i_plot, :].sum(1), unmasked_pix, nside=256), nest=True,
                        title="Counts: " + str(int(tot_counts_sel_DM_PS[i_plot].sum())) + ", Constraint: "
                              + str(int(100 * constraints_95_sel[i_plot])) + "%")
            zoom_ax(plt.gca())
            plt.tight_layout()

        print(lim95(sys_maps_sel[i_plot, :, :].sum(1).astype(int)))
        print("\n")
