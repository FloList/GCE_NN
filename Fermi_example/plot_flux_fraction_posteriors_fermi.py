"""
This script plots the flux fractions as estimated by the NNs for the Fermi map.
Results from run: Fermi_example_add_two_256_BN_bs_256_softplus_pre_gen
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as sp

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'CMRmap'
plt.ion()
sns.set_style("ticks")
sns.set_context("talk")

# Set means
model_names = [r"Diffuse $\pi^0$ + BS", "Diffuse IC", "Isotropic", r"$\it{Fermi}$ bubbles", "GCE", "Disk"]
colours = ['#ff0000', '#ec7014', '#fec44f', '#37c837', 'deepskyblue', 'k']
save_fig = False
means_full = np.asarray([0.5256651, 0.270391, 0.01978799, 0.06156372, 0.07928304, 0.04330921])  # means:
std_full = np.asarray([0.00744413, 0.01342631, 0.00916832, 0.00452482, 0.00478658, 0.01155945])  # pred. stds

# Plot settings
sns.set_context("talk")
sns.set_style("ticks")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 14
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')

# Make plot
fig, ax = plt.subplots(figsize=(6.44, 4.57))
std_fac = 4
lw = 2
do_fill = True
for i_model in range(len(model_names)):
    y_vec = np.linspace(means_full[i_model] - std_fac * std_full[i_model], means_full[i_model] + std_fac * std_full[i_model], 1000)
    if do_fill:
        ax.fill_between(100 * y_vec, sp.stats.norm.pdf(y_vec, means_full[i_model], std_full[i_model]), color=colours[i_model],
                        lw=lw, linestyle="-", alpha=0.175)
        ax.plot(100 * y_vec, sp.stats.norm.pdf(y_vec, means_full[i_model], std_full[i_model]), color=colours[i_model], lw=lw,
                linestyle="-", label=str(i_model))
    else:
        ax.plot(100 * y_vec, sp.stats.norm.pdf(y_vec, means_full[i_model], std_full[i_model]), color=colours[i_model], lw=lw, linestyle="-", label=str(i_model))

ax.set_xlim([0, 57])
ax.set_ylim([0, 100])
xticks = np.arange(0, 60, 10)
ax.set_xticks(xticks)
ax.set_xlabel(r"Flux fractions [$\%$]")
ax.set_ylabel("Probability density")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, model_names, ncol=1)
plt.tight_layout()

if save_fig:
    fig.savefig("Posterior_flux_fractions_fermi.pdf", bbox_inches="tight")
