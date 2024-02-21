from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import os
import sys
plt.ion()
import GCE.gce
import swyft

gce = GCE.gce.Analysis()

gce.load_params("../../parameter_files/Code_comparison/parameters_code_comparison.py")

gce.print_params()

# Ray settings (for parallelized data generation)
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000}
ray_settings = {"num_cpus": 4, "local_mode": False}  # select the number of CPUs here (local_mode=True for testing)
# gce.generate_template_maps(ray_settings, n_example_plots=5, job_id=0)

# gce.combine_template_maps(save_filenames=True, do_combine=True)

gce.build_pipeline()

# Manually test the generator
pair = next(gce.generators["test"].get_next_pair())
print(pair["data"].shape, pair["label"][0].shape, pair["label"][1].shape)

# import pickle
# with open("../data/Combined_maps/.....", 'rb') as f:
#     data = pickle.load(f)

samples = gce.datasets["test"].get_samples(1)
data, labels = samples["data"], samples["label"]  # samples contains data and labels (flux fractions & SCD histograms)
print("Shapes:")
print("  Data", data.shape)  # n_samples x n_pix_in_ROI x n_bins
print("  Flux", labels[0].shape)  # n_samples x n_templates x n_bins
print("  SCD histograms", labels[1].shape)  # n_samples x n_bins x n_PS_templates

# NOTE: the maps are stored in NEST format
# map_to_plot = 0
# lon_min = gce.p.data["lon_min"]
# lon_max = gce.p.data["lon_max"]
# lat_min = gce.p.data["lat_min"]
# lat_max = gce.p.data["lat_max"]
# hp.cartview(gce.decompress((data[map_to_plot] * gce.template_dict["rescale_compressed"]).sum(-1)), nest=True,
#             title="Simulated data: Count space", lonra=[lon_min, lon_max], latra=[lat_min, lat_max])
# hp.cartview(gce.decompress(data[map_to_plot].sum(-1)), nest=True,
#             title="Simulated data: Flux space", lonra=[lon_min, lon_max], latra=[lat_min, lat_max])
# hp.cartview(gce.decompress(gce.template_dict["rescale_compressed"].mean(-1), fill_value=np.nan), nest=True,
#             title="Fermi exposure correction", lonra=[lon_min, lon_max], latra=[lat_min, lat_max])
# plt.show()

gce.build_nn()

# gce.load_nn()
# gce.train_nn("flux_fractions")
# gce.train_nn("histograms")

n_samples = 20

pair = next(gce.generators["test"].get_next_pair())
print(pair["data"].shape, pair["label"][0].shape, pair["label"][1].shape)

test_samples = gce.datasets["test"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][1]
tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%
pred = gce.predict(test_data, tau=tau, multiple_taus=True)  # get the NN predictions

# Make some plots (will be saved in the models folder)
gce.plot_nn_architecture()
gce.plot_flux_fractions(test_ffs, pred)
gce.plot_histograms(test_hists, pred, plot_inds=np.arange(9))
gce.plot_maps(test_data, decompress=True, plot_inds=np.arange(9))
plt.show()
