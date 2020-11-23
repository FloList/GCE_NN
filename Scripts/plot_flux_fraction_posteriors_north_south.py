"""
This script plots the flux fractions as estimated by the NNs trained on the northern and southern hemispheres separately
and on both hemispheres simultaneously.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as sp
from gce_utils import pretty_plots

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'CMRmap'
plt.ion()
sns.set_style("white")
sns.set_context("talk")

# Set means
model_names = [r"diffuse $\pi^0$ + BS", "diffuse IC", "isotropic", r"$\it{Fermi}$ bubbles", "GCE DM", "GCE PS", "disk PS"]
colours = ['#ff0000', '#ec7014', '#fec44f', '#37c837', 'deepskyblue', 'darkslateblue', 'k']
save_fig = False
zoom = False  # zoom into the low flux region
means_full = np.asarray([0.5378, 0.2696, 0.0022104, 0.06837, 0.08569, 0.003224, 0.03312])  # means for the full map
std_full = np.asarray([0.0121, 0.01872, 0.005971, 0.006036, 0.01630, 0.01142, 0.01922])  # pred. stds for the full map
means_N = np.asarray([0.6297, 0.2243, 0.008635, 0.06805, 0.05170, 0.01909, 0.006262])  # N hemisphere
std_N = np.asarray([0.02205, 0.02358, 0.005153, 0.01193, 0.03106, 0.03069, 0.01154])
means_S = np.asarray([0.5280, 0.2835, 0.01415, 0.06739, 0.08086, 0.002334, 0.02382])  # S hemisphere
std_S = np.asarray([0.02325, 0.02748, 0.01357, 0.009052, 0.01697, 0.01278, 0.02668])

# Make plot
fig, ax = plt.subplots(figsize=(10, 8))
std_fac = 4
lw = 2
NS_fill = True
for i_model in range(len(model_names)):
    y_vec = np.linspace(means_full[i_model] - std_fac * std_full[i_model], means_full[i_model] + std_fac * std_full[i_model], 1000)
    if NS_fill:
        ax.fill_between(y_vec, sp.stats.norm.pdf(y_vec, means_full[i_model], std_full[i_model]), color=colours[i_model],
                        lw=lw, linestyle="-", alpha=0.175)
        ax.plot(y_vec, sp.stats.norm.pdf(y_vec, means_full[i_model], std_full[i_model]), color=colours[i_model], lw=lw,
                linestyle="-", label=str(i_model))
    else:
        ax.plot(y_vec, sp.stats.norm.pdf(y_vec, means_full[i_model], std_full[i_model]), color=colours[i_model], lw=lw, linestyle="-", label=str(i_model))
    y_vec = np.linspace(means_N[i_model] - std_fac * std_N[i_model], means_N[i_model] + std_fac * std_N[i_model], 1000)
    ax.plot(y_vec, sp.stats.norm.pdf(y_vec, means_N[i_model], std_N[i_model]), color=colours[i_model], lw=lw, linestyle="-.", label="N")
    y_vec = np.linspace(means_S[i_model] - std_fac * std_S[i_model], means_S[i_model] + std_fac * std_S[i_model], 1000)
    ax.plot(y_vec, sp.stats.norm.pdf(y_vec, means_S[i_model], std_S[i_model]), color=colours[i_model], lw=lw, linestyle="--", label="S")

ax.set_xlim([0, .7])
if zoom:
    ax.set_xlim([0, .15])
ax.set_ylim([0, 80])
ax.set_xlabel("Flux fractions")
ax.set_ylabel("Probability density")
handles, labels = ax.get_legend_handles_labels()
# if not zoom:
    # ax.legend(handles, model_names, ncol=2)

lines = ax.get_lines()
if not zoom:
    if NS_fill:
        legend1 = plt.legend(handles[::3], model_names, ncol=2)
        legend2 = plt.legend([lines[i] for i in [-3, -2, -1]], ["N+S", "N", "S"], loc=5)
    else:
        legend1 = plt.legend(handles[::3], model_names, ncol=2)
        legend2 = plt.legend([lines[i] for i in [-3, -2, -1]], ["N+S", "N", "S"], loc=5)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

pretty_plots()

if save_fig:
    if zoom:
        fig.savefig("N_S_posterior_flux_fractions_zoom.pdf", bbox_inches="tight")
    else:
        fig.savefig("N_S_posterior_flux_fractions.pdf", bbox_inches="tight")
