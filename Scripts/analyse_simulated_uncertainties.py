"""
This script can be used to simulate aleatoric and epistemic uncertainties and check that the coverage makes sense.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sys
from gce_utils import *
import os
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
plt.ion()

# Settings
n_points = 2500  # no. of data points
true_estimator_var = 0.04  # estimator is assumed to have no bias (i.e. E[mean] = 0), and this variance
mean_aleat_var = 0.02  # the mean amount of aleatoric uncertainty variance as estimated by the estimator
mean_epist_var = 0.2  # the mean amount of epistemic uncertainty variance in the estimator predictions
n_epist_samples = 30  # the number of samples to use for simulating the MC sampling
clip_min = -np.infty
clip_max = np.infty
do_plot = True

# Define "true" data
true_fluxes = np.linspace(0.0, 1.0, n_points)

# Generate "predicted" data
mean_estimates = np.clip(true_fluxes + np.random.normal(0.0, np.sqrt(true_estimator_var), n_points), clip_min, clip_max)  # simulate the mean predictions
MC_dropout_noise = np.random.normal(0.0, np.sqrt(mean_epist_var), [n_epist_samples, n_points])  # draw MC dropout epistemic uncertainty noise
MC_dropout_noise -= MC_dropout_noise.mean(0)  # remove mean in order not to affect the error of the simulator
pred_fluxes_MC = np.clip(mean_estimates[None] + MC_dropout_noise, clip_min, clip_max)  # simulate the MC dropout means
pred_fluxes_MC_aleat_var = np.random.chisquare(1, [n_epist_samples, n_points]) * mean_aleat_var  # simulate the MC dropout aleat. vars
pred_fluxes_mean = pred_fluxes_MC.mean(0)  # calculate the MC dropout means, this is equal to mean_estimates since MC_dropout_noise has mean 0
pred_fluxes_aleat_var = pred_fluxes_MC_aleat_var.mean(0)  # calculate the MC dropout means of the aleat. std
pred_mean_error = pred_fluxes_mean - true_fluxes  # calculate the error of the means
pred_fluxes_epist_var = pred_fluxes_MC.var(0, ddof=0)  # calculate the epistemic uncertainties

# Plot
if do_plot:
    fig, ax = plt.subplots(1, 3)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    ax[0].errorbar(x=true_fluxes, y=pred_fluxes_mean,
                         yerr=np.sqrt(pred_fluxes_aleat_var), fmt="o",
                         elinewidth=2, markersize=8, mfc="navy", mec="k", ecolor="navy")
    ax[0].set_title("Aleatoric uncertainty")
    ax[1].errorbar(x=true_fluxes, y=pred_fluxes_mean,
                         yerr=np.sqrt(pred_fluxes_epist_var), fmt="o",
                         elinewidth=2, markersize=8, mfc="navy", mec="k", ecolor="navy")
    ax[1].set_title("Epistemic uncertainty")
    ax[2].errorbar(x=true_fluxes, y=pred_fluxes_mean,
                         yerr=np.sqrt(pred_fluxes_epist_var + pred_fluxes_aleat_var), fmt="o",
                         elinewidth=2, markersize=8, mfc="navy", mec="k", ecolor="navy")
    ax[2].set_title("Predictive uncertainty")
    for ax_ in ax:
        ax_.set_xlabel("Truth")
        ax_.set_ylabel("Prediction")

# Print stats
print("Average predicted aleatoric uncertainty variance: {:2.3f}".format(pred_fluxes_aleat_var.mean()))
print("Average predicted epistemic uncertainty variance: {:2.3f}".format(pred_fluxes_epist_var.mean()))
print("Average total predicted uncertainty variance {:2.3f}".format((pred_fluxes_aleat_var + pred_fluxes_epist_var).mean()))
print("Average prediction error variance: {:2.3f}".format(np.var(pred_mean_error).mean()))

# Calculate coverages
print("Coverages:")
print(calculate_coverage(np.expand_dims(pred_fluxes_MC, -1), np.expand_dims(np.expand_dims(pred_fluxes_MC_aleat_var, -1), -1),
                   np.expand_dims(true_fluxes, -1)))
