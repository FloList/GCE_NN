from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import os
import sys
plt.ion()
import GCE.gce
import swyft

gce = GCE.gce.Analysis()

gce.load_params("../parameter_files/parameters.py")


gce.print_params()

# Ray settings (for parallelized data generation)
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000}
# ray_settings = {"num_cpus": 4}  # select the number of CPUs here
# gce.generate_template_maps(ray_settings, n_example_plots=5, job_id=0)


# gce.combine_template_maps(save_filenames=True, do_combine=True)

gce.build_pipeline()

# import pickle
# with open("../data/Combined_maps/Example_comb_128/Validation/Maps_00_val.pickle", 'rb') as f:
#     data = pickle.load(f)

samples = gce.datasets["val"].get_samples(1)
data, labels = samples["data"], samples["label"]  # samples contains data and labels (flux fractions & SCD histograms)
print("Shapes:")
print("  Data", data.shape)  # n_samples x n_pix_in_ROI
print("  Flux fractions", labels[0].shape)  # n_samples x n_templates
print("  SCD histograms", labels[1].shape)  # n_samples x n_bins x n_PS_templates

# NOTE: the maps are stored in NEST format
# map_to_plot = 0
# r = gce.p.data["outer_rad"] + 1
# hp.cartview(gce.decompress(data[map_to_plot] * gce.template_dict["rescale_compressed"]), nest=True,
#             title="Simulated data: Count space", lonra=[-r, r], latra=[-r, r])
# hp.cartview(gce.decompress(data[map_to_plot]), nest=True,
#             title="Simulated data: Flux space", lonra=[-r, r], latra=[-r, r])
# hp.cartview(gce.decompress(gce.template_dict["rescale_compressed"], fill_value=np.nan), nest=True,
#             title="Fermi exposure correction", lonra=[-r, r], latra=[-r, r])
# plt.show()
#
# fermi_counts = gce.datasets["test"].get_fermi_counts()
# r = gce.p.data["outer_rad"] + 1
# hp.cartview(gce.decompress(fermi_counts * gce.generators["test"].settings_dict["rescale_compressed"]), nest=True,
#             title="Fermi data: Count space", max=100, lonra=[-r, r], latra=[-r, r])
# # hp.cartview(gce.decompress(fermi_counts), nest=True, title="Fermi data: Flux space", max=100)
# plt.show()

gce.build_nn()

# gce.load_nn()
gce.delete_run(confirm=False)
gce.train_nn("flux_fractions")
# gce.train_nn("histograms")

n_samples = 20
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
