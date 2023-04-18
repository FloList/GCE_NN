from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import os
import sys
plt.ion()
import GCE.gce

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
gce.train_nn("histograms")
# sys.exit(0)
# gce.load_nn()

n_samples = 5
test_samples = gce.datasets["test"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][1]
tau = np.asarray([0.003, 0.05, 0.16, 0.5, 0.84, 0.95, 0.997])
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
