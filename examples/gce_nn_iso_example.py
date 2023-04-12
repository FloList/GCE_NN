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
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000, "local_mode": False}
# ray_settings = {"num_cpus": 4}  # select the number of CPUs here
# gce.generate_template_maps(ray_settings, n_example_plots=25, job_id=0)
# gce.combine_template_maps(save_filenames=True, do_combine=True)
# sys.exit(0)

# gce.delete_run(confirm=False)
gce.build_pipeline()
gce.build_nn()
gce.train_nn("histograms")
# gce.load_nn()

n_samples = 5
test_samples = gce.datasets["test"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][1]
tau = np.asarray([0.003, 0.05, 0.16, 0.5, 0.84, 0.95, 0.997])
tau_mid = (tau[1:] + tau[:-1]) / 2.0
normed_flux_queries = np.arange(0.01, 1.0, 0.01)

preds = []
for f_query in normed_flux_queries:
    preds.append(gce.predict(test_data, tau=tau, multiple_taus=True, normed_flux_queries=np.tile(np.atleast_2d(f_query), (n_samples, 1))))  # get the NN predictions

hists = np.asarray([p["hist"] for p in preds])

colors = plt.cm.get_cmap("RdBu")(tau_mid)
median_ind = len(tau) // 2
fig, axs = plt.subplots(1, n_samples)
for i_sample in range(n_samples):
    axs[i_sample].plot(test_hists[i_sample, :, 0].cumsum(), "k-", zorder=3)
    for i_tau in range(len(tau) - 1):
        axs[i_sample].fill_between(10000 * normed_flux_queries,
                                   y1=hists[:, i_tau, i_sample, 0, 0],
                                   y2=hists[:, i_tau+1, i_sample, 0, 0],
                                   color=colors[i_tau], alpha=0.5)
    axs[i_sample].plot(10000 * normed_flux_queries, hists[:, median_ind, i_sample, 0, 0], color="gold")
