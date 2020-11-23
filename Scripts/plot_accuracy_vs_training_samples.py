"""
This script plots the mean accuracy as a function of the number of training samples used.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gce_utils import pretty_plots

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.cmap'] = 'CMRmap'
plt.ion()
sns.set_style("ticks")
sns.set_context("talk")

# Set means
model_names = [r"diffuse $\pi^0$ + BS", "diffuse IC", "isotropic", r"$\it{Fermi}$ bubbles", "GCE DM", "GCE PS", "disk PS"]
colours = ['#ff0000', '#ec7014', '#fec44f', '#37c837', 'deepskyblue', 'darkslateblue', 'k']
markers = ["o", "v", "^", "s", "d", "H", "X"]
save_fig = False

training_samples = [6000, 19000, 60000, 190000, 600000]
mean_errors = np.asarray([[1.38, 2.04, 1.03, 0.66, 1.99, 2.12, 1.81],
                          [1.06, 1.75, 0.91, 0.51, 1.70, 1.75, 1.54],
                          [0.98, 1.76, 0.91, 0.51, 1.70, 1.75, 1.54],
                          [0.84, 1.50, 0.77, 0.42, 1.60, 1.69, 1.32],
                          [0.82, 1.51, 0.75, 0.40, 1.54, 1.63, 1.30]])

# Make plot
fig, ax = plt.subplots(figsize=(10, 8))
lw = 2
for i_model in range(len(model_names)):
    ax.plot(training_samples, mean_errors[:, i_model], color=colours[i_model], lw=lw, marker=markers[i_model])
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_ylabel("Mean error [%]")
ax.set_xlabel("Number of training maps")
pretty_plots()
ax.set_ylim(0, 2.75)
ax.legend(model_names, ncol=2)
# ax.axes.yaxis.set_tick_params(length=0, pad=10)
sns.despine(fig, offset=0)


if save_fig:
    fig.savefig("mean_accuracy_wrt_training_samples.pdf", bbox_inches="tight")
