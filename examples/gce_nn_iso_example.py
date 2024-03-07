from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import os
import sys
plt.ion()
import GCE.gce
from scipy.interpolate import interp1d
from scipy.stats import norm
gce = GCE.gce.Analysis()

gce.load_params("../parameter_files/parameters_isotropic.py")
gce.print_params()

# Ray settings (for parallelized data generation)
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000, "local_mode": True}
# ray_settings = {"num_cpus": 4}  # select the number of CPUs here
# gce.generate_template_maps(ray_settings, n_example_plots=5, job_id=0)
# gce.combine_template_maps(save_filenames=True, do_combine=True)
# sys.exit(0)

# gce.delete_run(confirm=False)
gce.build_pipeline()
gce.build_nn()
gce.load_nn()
gce.train_nn("histograms", new_optimizer=True)
# sys.exit(0)
# gce.load_nn()

n_samples = 5
test_samples = gce.datasets["test"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][1]
tau = np.asarray([0.003, 0.05, 0.16, 0.5, 0.84, 0.95, 0.997])
n_taus = len(tau)
tau_mid = (tau[1:] + tau[:-1]) / 2.0
normed_flux_queries = np.arange(0.01, 1.0, 0.01)

all_tau = np.tile(np.repeat(tau, len(normed_flux_queries)), n_samples)
all_normed_flux_queries = np.tile(np.tile(normed_flux_queries, len(tau)), n_samples)
preds = gce.predict(test_data, tau=all_tau[:, None], normed_flux_queries=all_normed_flux_queries[:, None])
hists = preds["hist"][:, 0, 0].numpy().reshape(n_samples, len(tau), len(normed_flux_queries))

colors = plt.cm.get_cmap("RdBu")(tau_mid)
median_ind = len(tau) // 2
x_vals_all = gce.p.nn.hist["nn_hist_centers"]
x_vals_plot = np.interp(normed_flux_queries, np.linspace(0.0, 1.0, gce.p.nn.hist.n_bins), x_vals_all)

fig, axs = plt.subplots(2, n_samples, figsize=(34, 12), squeeze=False)
for i_sample in range(n_samples):
    # Plot dN/dF
    axs[0, i_sample].semilogx(x_vals_all, test_hists[i_sample, :, 0].cumsum(), "k-", zorder=3)
    for i_tau in range(len(tau) - 1):
        axs[0, i_sample].fill_between(x_vals_plot,
                                   y1=hists[i_sample, i_tau, :], y2=hists[i_sample, i_tau + 1, :],
                                   color=colors[i_tau], alpha=0.5)
    axs[0, i_sample].semilogx(x_vals_plot, hists[i_sample, median_ind, :], color="gold")
    axs[0, i_sample].set_xlabel(r"$F$")
    axs[0, i_sample].axvline(1, color="k", ls="--")

    # Plot map
    hp.cartview(gce.decompress(test_data[i_sample], fill_value=np.nan), nest=True, fig=fig,
                sub=(2, n_samples, n_samples + i_sample + 1), title=f"{i_sample}", cmap="magma")
    axs[1, i_sample].axis("off")


axs[0, 0].set_ylabel(r"$F^2 \ \frac{dN}{dF}$")
plt.tight_layout()
plt.subplots_adjust(hspace=0)

# Check monotonicity
min_diff_wrt_F = np.min(np.diff(hists, axis=2))  # monotonicity w.r.t. F
min_diff_wrt_tau = np.min(np.diff(hists, axis=1))  # monotonicity w.r.t. tau

# Compute coverage  # TODO!!!
assert np.all(np.abs(tau + tau[::-1] - 1.0) < 1e-8), "Quantile levels must be symmetric for coverage check!"
cum_hists = test_hists.cumsum(axis=1)
eps_tol = 1e-5  # exclude fluxes for which the true CDF lies outside [eps_tol, 1 - eps_tol]
coverages = np.zeros(n_taus // 2)
alphas = (tau[::-1] - tau)[:(n_taus // 2)]
n_alphas = len(alphas)

indices_pred_in_true = np.floor(normed_flux_queries * test_hists.shape[1]).astype(int)
for i_alpha in range(n_alphas):
    cov_vec_for_alpha = (hists[:, i_alpha, :] <= cum_hists[:, indices_pred_in_true, 0]) \
                         & (hists[:, n_taus - (i_alpha + 1), :] >= cum_hists[:, indices_pred_in_true, 0])
    relevant_for_alpha = (cum_hists[:, indices_pred_in_true, 0] > eps_tol) & ((1 - cum_hists[:, indices_pred_in_true, 0]) > eps_tol)
    cov_val_for_alpha = cov_vec_for_alpha.flatten()[relevant_for_alpha.flatten()].mean()
    coverages[i_alpha] = cov_val_for_alpha

print("Nominal:", alphas)
print("Empirical coverage:", coverages)

# Now: apply Bonferroni correction to get simultaneous prediction intervals!
def generate_random_variable(n_samples, tau, quantiles):
    # Sort the quantiles in ascending order
    sort_idx = np.argsort(quantiles)
    quantiles = np.asarray(quantiles)[sort_idx]
    tau = np.array(tau)[sort_idx]
    # Add 0 and 1 to tau
    tau = np.concatenate([[0], tau, [1]])
    # Add min. and max. again to quantiles  # TODO: Add values of neighbouring bins!
    quantiles = np.concatenate([[quantiles[0]], quantiles, [quantiles[1]]])
    # Generate uniform random variables
    U = np.random.uniform(size=n_samples)
    # Find the indices of the quantiles corresponding to the uniform random variables
    idx = np.searchsorted(tau, U, side="right").astype(int)
    # Interpolate between adjacent quantiles to find the corresponding values of X
    lower_cdf_vals = tau[idx - 1]
    upper_cdf_vals = tau[idx]
    lower_quantiles = quantiles[idx - 1]
    upper_quantiles = quantiles[idx]
    X = interp1d(np.hstack((lower_cdf_vals, upper_cdf_vals)), np.hstack((lower_quantiles, upper_quantiles)))(U)
    return X


bonferroni_sample = gce.datasets["test"].get_samples(1)
# tau_bonferroni = norm.cdf(np.linspace(-3, 3, 100))
tau_bonferroni = np.linspace(0.01, 0.99, 99)
n_taus_bonferroni = len(tau_bonferroni)
normed_flux_queries_bonferroni = np.arange(0.05, 1.0, 0.05)
all_tau_bonferroni = np.repeat(tau_bonferroni, len(normed_flux_queries_bonferroni))
all_normed_flux_queries_bonferroni = np.tile(normed_flux_queries_bonferroni, len(tau_bonferroni))
preds_bonferroni = gce.predict(bonferroni_sample["data"], tau=all_tau_bonferroni[:, None], normed_flux_queries=all_normed_flux_queries_bonferroni[:, None])
hists_bonferroni = preds_bonferroni["hist"][:, 0, 0].numpy().reshape(1, len(tau_bonferroni), len(normed_flux_queries_bonferroni))

n_samples_bonferroni = 1000
bootstrap_samples = np.apply_along_axis(lambda x: generate_random_variable(n_samples=n_samples_bonferroni, tau=tau_bonferroni, quantiles=x), axis=0, arr=hists_bonferroni[0])
bootstrap_samples = np.sort(bootstrap_samples, axis=0)

colors_bonferroni = plt.cm.get_cmap("RdBu")(tau_bonferroni)
fig, ax = plt.subplots(1, 1)
ax.plot(bootstrap_samples.T, ls="none", marker=".")

median_ind_bonferroni = len(tau_bonferroni) // 2
tau_pos_bonferroni = tau_bonferroni[(n_taus_bonferroni // 2 + 1):]
n_tau_pos_bonferroni = len(tau_pos_bonferroni)
adj_tau_pos_bonferroni = 1 - (((1 - tau_pos_bonferroni) / len(normed_flux_queries_bonferroni)) / 2)
lower_bounds = np.quantile(bootstrap_samples, q=1 - adj_tau_pos_bonferroni, axis=0)
upper_bounds = np.quantile(bootstrap_samples, q=adj_tau_pos_bonferroni, axis=0)
adj_hists_bonferroni = np.concatenate([lower_bounds, hists_bonferroni[0, median_ind_bonferroni:median_ind_bonferroni + 1, :], upper_bounds], axis=0)
x_vals_plot_bonferroni = np.interp(normed_flux_queries_bonferroni, np.linspace(0.0, 1.0, gce.p.nn.hist.n_bins), x_vals_all)

# Plot pointwise and simultaneous prediction intervals
fig, axs = plt.subplots(1, 2)
axs[0].semilogx(x_vals_all, bonferroni_sample["label"][1][0, :, 0].cumsum(), "k-", zorder=3)
# PW
for i_tau in range(len(tau_bonferroni) - 1):
    axs[0].fill_between(x_vals_plot_bonferroni, y1=hists_bonferroni[0, i_tau, :], y2=hists_bonferroni[0, i_tau + 1, :],
                                  color=colors_bonferroni[i_tau], alpha=0.5)
axs[0].semilogx(x_vals_plot_bonferroni, hists_bonferroni[0, median_ind_bonferroni, :], color="gold")
axs[0].set_xlabel(r"$F$")
axs[0].axvline(1, color="k", ls="--")
axs[0].set_title("Pointwise prediction intervals")

# Simultaneous
axs[1].semilogx(x_vals_all, bonferroni_sample["label"][1][0, :, 0].cumsum(), "k-", zorder=3)
for i_tau in range(len(tau_bonferroni) - 1):
    axs[1].fill_between(x_vals_plot_bonferroni, y1=adj_hists_bonferroni[i_tau, :], y2=adj_hists_bonferroni[i_tau + 1, :],
                        color=colors_bonferroni[i_tau], alpha=0.5)
axs[1].semilogx(x_vals_plot_bonferroni, adj_hists_bonferroni[median_ind_bonferroni, :], color="gold")
axs[1].set_xlabel(r"$F$")
axs[1].axvline(1, color="k", ls="--")
axs[1].set_title("Simultaneous prediction intervals")