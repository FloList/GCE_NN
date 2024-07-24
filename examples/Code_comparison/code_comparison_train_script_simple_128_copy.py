from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import healpy as hp
import os
import sys
plt.ion()
import GCE.gce

gce = GCE.gce.Analysis()

gce.load_params("../../parameter_files/Code_comparison/parameters_code_comparison_simple_128_copy.py")

gce.print_params()

# Ray settings (for parallelized data generation)
# ray_settings = {"num_cpus": 4, "object_store_memory": 2000000000}
# ray_settings = {"num_cpus": 16, "local_mode": False}  # select the number of CPUs here (local_mode=True for testing)
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
# # hp.cartview(gce.decompress(gce.template_dict["rescale_compressed"].mean(-1), fill_value=np.nan), nest=True,
# #             title="Fermi exposure correction", lonra=[lon_min, lon_max], latra=[lat_min, lat_max])
# plt.show()

gce.build_nn()


# gce.load_nn()
# gce.delete_run(confirm=False)
gce.train_nn("flux_fractions")
gce.train_nn("histograms")

n_samples = 50

# pair = next(gce.generators["test"].get_next_pair())
# print(pair["data"].shape, pair["label"][0].shape, pair["label"][1].shape)

test_samples = gce.datasets["test"].get_samples(n_samples)
test_data, test_ffs, test_hists = test_samples["data"], test_samples["label"][0], test_samples["label"][1]
tau = np.arange(5, 100, 5) * 0.01  # quantile levels for SCD histograms, from 5% to 95% in steps of 5%
pred = gce.predict(test_data, tau=tau, multiple_taus=True)  # get the NN predictions
pred["ff_mean_full"] = pred["ff_mean"]    # store the full predictions
pred["ff_logvar_full"] = pred["ff_logvar"]
pred["ff_mean"] = tf.reduce_mean(pred["ff_mean_full"], axis=-1)  # average over energy bins
pred["ff_logvar"] = tf.reduce_mean(pred["ff_logvar_full"], axis=-1)
test_data_sum_E = test_data.sum(-1)  # average over energy bins

# Make some plots (will be saved in the models folder)
# gce.plot_nn_architecture()
labels_samples = test_ffs.sum(-1) / test_ffs.sum(-1).sum(-1, keepdims=True)
gce.plot_flux_fractions(labels_samples, pred)
plt.gcf().savefig("test_data_flux_fractions.pdf", bbox_inches="tight")
gce.plot_histograms(test_hists, pred, plot_inds=np.arange(9))
plt.gcf().savefig("test_data_hist.pdf", bbox_inches="tight")

gce.plot_histograms(test_hists, pred, plot_inds=np.arange(9), pdf=True)
for ax in plt.gcf().get_axes():
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(1e-13, 8e-9)

gce.plot_maps(test_data_sum_E, decompress=True, plot_inds=np.arange(9))
plt.show()
plt.gcf().savefig("test_data_maps.pdf", bbox_inches="tight")

# Now, load the benchmark maps
maps_nside_256_full = np.array(
    [np.load(os.path.join(gce.p.gen["fermi_root"], f"count_map_{i}_256.npy")) for i in range(1, 4)])
print("Total number of counts: ", maps_nside_256_full.sum(-1).sum(-1))
maps_nside_128_full_nest = np.array([hp.ud_grade(maps_nside_256_full[i], nside_out=128, order_in="RING", order_out="NEST",
                                            power=-2, dtype=np.float32) for i in range(3)])

# Compress the maps to mask
maps_nside_128_nest = np.array([gce.compress(maps_nside_128_full_nest[i]) for i in range(3)])
print("Total number of counts in ROI: ", maps_nside_128_nest.sum(-1).sum(-1))

# Remove exposure correction
maps_nside_128_nest_transposed = np.transpose(maps_nside_128_nest, (0, 2, 1))
maps_final = maps_nside_128_nest_transposed / gce.template_dict["rescale_compressed"][None, :, :]

# Get the predictions
pred_benchmark = gce.predict(maps_final, tau=tau, multiple_taus=True)
pred_benchmark["ff_mean_full"] = pred_benchmark["ff_mean"]
pred_benchmark["ff_mean"] = tf.reduce_mean(pred_benchmark["ff_mean_full"], axis=-1)

# Also load the flux lists
flux_arrs = [np.load(os.path.join(gce.p.gen["fermi_root"], f"source_info_{i}.npz"))["flux"] for i in range(1, 4)]
flux_hists = np.array([np.histogram(flux_arrs[i] * 1e-4, bins=gce.p.nn.hist["nn_hist_bins"], weights=flux_arrs[i])[0] for i in range(3)])  # this is in 1/(m^2 s)
flux_hists /= flux_hists.sum(-1, keepdims=True)
gce.plot_histograms(flux_hists[:, :, None], pred_benchmark, plot_inds=np.arange(3))
plt.gcf().savefig("benchmark_results_cdf.pdf", bbox_inches="tight")

gce.plot_histograms(flux_hists[:, :, None], pred_benchmark, plot_inds=np.arange(3), pdf=True)
plt.gcf().savefig("benchmark_results_pdf.pdf", bbox_inches="tight")




# Make some plots
# gce.plot_histograms(None, pred_benchmark)
# plt.gcf().savefig("benchmark_hist_cdf.pdf", bbox_inches="tight")
#
# gce.plot_histograms(None, pred_benchmark, pdf=True)
# for ax in plt.gcf().get_axes():
#     # ax.set_yscale("linear")
#     # ax.set_ylim(1e-4, 1)
#     ax.set_xlim(1e-13, 8e-9)
# plt.gcf().savefig("benchmark_hist_pdf.pdf", bbox_inches="tight")


def dNdF(F, A, Fb, n1, n2):
    if F >= Fb:
        return A * (F / Fb) ** (-n1)
    else:
        return A * (F / Fb) ** (-n2)

dNdF = np.vectorize(dNdF)


def CDF(F, Fb, n1, n2):
    if F < Fb:
        return (n1 - 1.) / (n1 - n2) * (F / Fb) ** (1. - n2)
    else:
        return 1. - (1. - n2) / (n1 - n2) * (F / Fb) ** (1. - n1)

CDF = np.vectorize(CDF)

# Parameters (guessed by Nick)
Ag = np.array([8.e10, 8.5e9, 3.1e12])
Fbg = np.array([2.e-11, 1.e-10, 1.e-12])
n1g = np.array([2.06, 2.8, 1.9])
n2g = np.array([-0.5, -0.1, -1.7])
N_tot = Ag * Fbg * (1 / (n1g - 1) + 1 / (1 - n2g))


# Plot the true F^2 dN/dF
from scipy.integrate import cumtrapz
F_eval = np.logspace(-18, -3, 10000)
dNdF_eval = np.array([dNdF(F_eval, Ag[i], Fbg[i], n1g[i], n2g[i]) for i in range(3)])
CDF_eval = np.array([CDF(F_eval, Fbg[i], n1g[i], n2g[i]) for i in range(3)])
colors = plt.cm.coolwarm(np.linspace(0, 1, len(tau)))

fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15, 5))
for i in range(3):
    truth = CDF_eval[i]
    # truth = dNdF_eval[i]
    # truth /= np.trapz(truth, F_eval)
    # truth_cum = cumtrapz(truth, F_eval, initial=0)
    axs[i].semilogx(F_eval, truth, label=f"True $dN/dF$ {i + 1}", color="k")
    # axs[i].loglog(F_eval, truth, label=f"True $dN/dF$ {i + 1}")

    for j in range(len(tau)):
        pred = pred_benchmark["hist"].numpy()[j, i, :, 0] * gce.p.nn.hist["nn_hist_centers"] ** 2  #/ gce.p.nn.hist["nn_hist_widths"] ** 2  # TODO: WHY SQUARE?
        pred_norm = np.trapz(pred, gce.p.nn.hist["nn_hist_centers"])
        pred /= pred_norm
        pred_cum = cumtrapz(pred, gce.p.nn.hist["nn_hist_centers"], initial=0)

        # pred /= pred.sum()
        # pred_cum = pred.cumsum()
        axs[i].semilogx(gce.p.nn.hist["nn_hist_centers"], pred_cum, color=colors[j], alpha=0.5)
        # axs[i].loglog(gce.p.nn.hist["nn_hist_centers"], pred * gce.p.nn.hist["nn_hist_centers"] ** 2, color=colors[j], alpha=0.5)
    # axs[i].legend()
    axs[i].set_xlabel(r"Flux [ph / cm$^2$ / s]")
    axs[i].set_ylabel(r"$CDF$")
    axs[i].set_xlim(1e-13, 1e-8)
plt.gcf().savefig("benchmark_hist_comparison.pdf", bbox_inches="tight")



# Load the first template map (pickle file)
import pickle
filename_iso_PS = os.path.join(gce.p.gen["template_maps_root"], gce.p.tt["data_name"] + "_" + str(gce.p.data["nside"]), "iso_PS", gce.p.tt["filename_base"] + "_0_0.pickle")
with open(filename_iso_PS, "rb") as f:
    template_map = pickle.load(f)
t_info = template_map["info"]
t_flux_arr = template_map["flux_arr"]
t_n_phot = template_map["n_phot"]
t_data = template_map["data"]

# Recover the true dNdF from the flux array
from scipy.stats import skewnorm
colors = plt.cm.Spectral(np.linspace(0, 1, 10))
plt.figure()

# gce.p.nn.hist["nn_hist_bins_orig"] = np.copy(gce.p.nn.hist["nn_hist_bins"])
# gce.p.nn.hist["nn_hist_bins"] = gce.p.nn.hist["nn_hist_bins_orig"]

for i, ind in enumerate(range(9)):
    t_flux_arr_hist = np.histogram(t_flux_arr[ind], bins=gce.p.nn.hist["nn_hist_bins"], weights=t_flux_arr[ind])[0]
    t_flux_arr_hist_norm = t_flux_arr_hist / t_flux_arr_hist.sum()

    t_true_dNdF_fun = lambda a, mu, sigma2, F: skewnorm(a, mu, sigma2 ** 0.5).pdf(F)
    t_true_dNdF = t_true_dNdF_fun(t_info["skew"][ind], t_info["means"][ind], t_info["vars"][ind], np.log10(F_eval))
    t_true_dNdF_cum = cumtrapz(t_true_dNdF, np.log10(F_eval), initial=0.0)

    # Normalization:
    print(np.trapz(t_true_dNdF / F_eval, F_eval) / np.log(10))  # this is ~one if entire PDF is within flux region of interest

    # Now, restore the dN/dF from the histogram
    t_dNdF_restored = t_flux_arr_hist_norm / gce.p.nn.hist["nn_hist_widths"] ** 2
    # t_dNdF_restored = t_dNdF_restored / (np.trapz(t_dNdF_restored, gce.p.nn.hist["nn_hist_centers"]) / np.log(10))
    t_dNdF_restored /= np.trapz(t_dNdF_restored, gce.p.nn.hist["nn_hist_centers"])
    t_dNdF_restored_cum = cumtrapz(t_dNdF_restored, gce.p.nn.hist["nn_hist_centers"], initial=0)

    # plt.xlim(1e-13, 1e-8)
    # plt.semilogx(F_eval, t_true_dNdF_cum, label="True $dN/dF$", color=colors[ind], ls="-")
    # plt.semilogx(gce.p.nn.hist["nn_hist_centers"], t_dNdF_restored, label="Restored $dN/dF$",
    #              color=colors[ind], ls="--")

    # t_dNdF_restored = t_dNdF_restored * gce.p.nn.hist["nn_hist_centers"]
    # print(np.trapz(t_dNdF_restored / gce.p.nn.hist["nn_hist_centers"], gce.p.nn.hist["nn_hist_centers"]) / np.log(10))

    plt.xlim(1e-13, 1e-8)
    plt.semilogx(F_eval, cumtrapz(t_true_dNdF, np.log10(F_eval), initial=0.0), label="True $dN/dF$", color=colors[ind], ls="-")
    plt.semilogx(gce.p.nn.hist["nn_hist_centers"], t_dNdF_restored_cum, label="Restored $dN/dF$",
                 color=colors[ind], ls="--")
    # plt.semilogx(F_eval, t_true_dNdF.cumsum() / t_true_dNdF.sum(), label="True CDF", color=colors[ind], ls="-")
    # plt.semilogx(gce.p.nn.hist["nn_hist_centers"], t_flux_arr_hist_norm.cumsum() / t_flux_arr_hist_norm.sum(), label="Restored CDF",
    #              color=colors[ind], ls="--")
plt.ylim(0, 1.1)
plt.savefig("template_comparison_dNdF_cum.pdf", bbox_inches="tight")























# Normalization
total_flux = (maps_nside_128_nest_transposed / gce.template_dict["exp_compressed"][None, :, :]).sum(-1).sum(-1)
total_flux_PS = total_flux * pred_benchmark["ff_mean"][:, 1]
total_sky_area_in_deg2 = 4 * np.pi * (180 / np.pi)**2
sky_area_in_roi_in_deg2 = maps_nside_128_nest.shape[-1] * total_sky_area_in_deg2 / hp.nside2npix(gce.p.data["nside"])
assert np.allclose(*[np.trapz(pred_benchmark["hist"].numpy()[0, i, :, 0], np.log(gce.p.nn.hist["nn_hist_centers"])) for i in range(3)])
current_norm_fac = 1.0  #np.trapz(pred_benchmark["hist"].numpy()[0, 0, :, 0], np.log(gce.p.nn.hist["nn_hist_centers"]))
F2dNdF = pred_benchmark["hist"].numpy()[:, :, :, 0] / sky_area_in_roi_in_deg2 * total_flux_PS[None, :, None].numpy() / current_norm_fac

# Compute the 1ph-line
f_1ph = 1 / gce.template_dict["exp_compressed"].mean()

# Plot the differential flux
n_tau = len(tau)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(tau)))

# True parameters (from Nick's plot)
from scipy.interpolate import interp1d
F_eval = np.logspace(-13, -6, 10000)

p1_x = np.array([3.2e-13, 2e-11, 1e-8])
p1_y = np.array([1e-15, 3.15e-11, 2.18e-11])

p2_x = np.array([4.5e-13, 1e-10, 1e-8])
p2_y = np.array([1e-15, 8.5e-11, 2.2e-12])

p3_x = np.array([1.2e-13, 1e-12, 1e-8])
p3_y = np.array([1e-15, 3.2e-12, 8.1e-12])

# Interpolate logarithmically
interp_p1 = np.exp(interp1d(np.log(p1_x), np.log(p1_y), kind="linear", fill_value="extrapolate")(np.log(F_eval)))
interp_p2 = np.exp(interp1d(np.log(p2_x), np.log(p2_y), kind="linear", fill_value="extrapolate")(np.log(F_eval)))
interp_p3 = np.exp(interp1d(np.log(p3_x), np.log(p3_y), kind="linear", fill_value="extrapolate")(np.log(F_eval)))

plt.figure()
plt.loglog(F_eval, interp_p1, label=r"True $F^2 dN/dF$ 1", color="firebrick", linestyle="--")
plt.loglog(F_eval, interp_p2, label=r"True $F^2 dN/dF$ 2", color="purple", linestyle="--")
plt.loglog(F_eval, interp_p3, label=r"True $F^2 dN/dF$ 3", color="darkorange", linestyle="--")
plt.scatter(p1_x, p1_y, color="firebrick", marker="x")
plt.scatter(p2_x, p2_y, color="purple", marker="x")
plt.scatter(p3_x, p3_y, color="darkorange", marker="x")
plt.legend()
plt.gcf().savefig("benchmark_hist_true_F2_dN_dF.pdf", bbox_inches="tight")

# Generate the "true" labels by evaluating the true dN/dF at the bin centers
# F2dNdF_true_p1 = np.exp(interp1d(np.log(F_eval), np.log(interp_p1), kind="linear", fill_value="extrapolate")(np.log(gce.p.nn.hist["nn_hist_centers"])))
# F2dNdF_true_p2 = np.exp(interp1d(np.log(F_eval), np.log(interp_p2), kind="linear", fill_value="extrapolate")(np.log(gce.p.nn.hist["nn_hist_centers"])))
# F2dNdF_true_p3 = np.exp(interp1d(np.log(F_eval), np.log(interp_p3), kind="linear", fill_value="extrapolate")(np.log(gce.p.nn.hist["nn_hist_centers"])))
F2dNdF_true_p1 = interp_p1
F2dNdF_true_p2 = interp_p2
F2dNdF_true_p3 = interp_p3

# Everything beyond 1e-8 has been cut it seems -> set to 0
cutoff = 1e-8
# F2dNdF_true_p1[gce.p.nn.hist["nn_hist_centers"] > cutoff] = 0
# F2dNdF_true_p2[gce.p.nn.hist["nn_hist_centers"] > cutoff] = 0
# F2dNdF_true_p3[gce.p.nn.hist["nn_hist_centers"] > cutoff] = 0
F2dNdF_true_p1[F_eval > cutoff] = 0
F2dNdF_true_p2[F_eval > cutoff] = 0
F2dNdF_true_p3[F_eval > cutoff] = 0

F2dNdF_true = np.array([F2dNdF_true_p1, F2dNdF_true_p2, F2dNdF_true_p3])


# Get the true cumulative labels by integrating the true dN/dF
# F2dNdF_true_p1_cum = np.trapz(F2dNdF_true_p1, np.log(gce.p.nn.hist["nn_hist_centers"]))
# F2dNdF_true_p2_cum = np.trapz(F2dNdF_true_p2, np.log(gce.p.nn.hist["nn_hist_centers"]))
# F2dNdF_true_p3_cum = np.trapz(F2dNdF_true_p3, np.log(gce.p.nn.hist["nn_hist_centers"]))

FdNdF = F2dNdF / gce.p.nn.hist["nn_hist_widths"]

fig, axs = plt.subplots(2, 3, figsize=(15, 5), constrained_layout=True)
for j in range(3):
    for i in range(n_tau):
        axs[0, j].loglog(gce.p.nn.hist["nn_hist_centers"], FdNdF[i, j] / FdNdF[i, j].sum(), color=colors[i], alpha=0.5)
        # axs[0, j].loglog(gce.p.nn.hist["nn_hist_centers"], F2dNdF_true[j], color="black", linestyle="--")
        axs[0, j].loglog(F_eval, F2dNdF_true[j], color="black", linestyle="--")
        axs[0, j].set_title(f"PS template {j}")
        axs[1, j].semilogx(gce.p.nn.hist["nn_hist_centers"], FdNdF[i, j].cumsum() / FdNdF[i, j].sum(), color=colors[i], alpha=0.5)
        # axs[1, j].semilogx(gce.p.nn.hist["nn_hist_centers"], F2dNdF_true[j].cumsum(), color="black", linestyle="--")
        axs[1, j].semilogx(F_eval, F2dNdF_true[j].cumsum(), color="black", linestyle="--")
    for d in range(2):
        axs[d, j].axvline(f_1ph, color="silver", linestyle="--")
        axs[d, j].set_title(f"Map {j}")
        axs[d, j].set_xlim(1e-13, 8e-9)
    axs[1, j].set_xlabel(r"Flux [ph / cm$^2$ / s]")
    axs[0, j].set_ylim(1e-15, 1e-9)

    if j == 0:
        axs[0, j].set_ylabel(r"$F^2 \, dN / dF$ [ph / cm$^2$ / s / deg$^2$]")
        axs[1, j].set_ylabel(r"Cumulative")

plt.gcf().savefig("benchmark_hist_pdf.pdf", bbox_inches="tight")

# plt.figure()
# for i in range(3):
#     plt.plot(F2dNdF_true[i].cumsum(), label=f"dNdF {i}", linestyle="--")
# plt.gcf().savefig("benchmark_hist_true_cdf_normalized.pdf", bbox_inches="tight")

# F2dNdF_true_normalized = F2dNdF_true
# F2dNdF_true_normalized /= F2dNdF_true_normalized.sum(-1, keepdims=True)
# gce.plot_histograms(F2dNdF_true_normalized[:, :, None], pred_benchmark, plot_inds=np.arange(3))
# plt.gcf().savefig("benchmark_hist_cdf_normalized.pdf", bbox_inches="tight")

gce.plot_histograms(None, pred_benchmark, plot_inds=np.arange(3))


def dNdF(F, A, Fb, n1, n2):
    if F >= Fb:
        return A * (F / Fb) ** (-n1)
    else:
        return A * (F / Fb) ** (-n2)

dNdF = np.vectorize(dNdF)


def CDF(F, Fb, n1, n2):
    if F < Fb:
        return (n1 - 1.) / (n1 - n2) * (F / Fb) ** (1. - n2)
    else:
        return 1. - (1. - n2) / (n1 - n2) * (F / Fb) ** (1. - n1)

CDF = np.vectorize(CDF)

# Parameters (guessed by Nick)
Ag = [8.e10, 8.5e9, 3.1e12]
Fbg = [2.e-11, 1.e-10, 1.e-12]
n1g = [2.06, 2.8, 1.9]
n2g = [-0.5, -0.1, -1.7]


gce.plot_histograms(None, pred_benchmark, plot_inds=np.arange(3))

Fv = np.logspace(-18, -8, 1000)
for i in range(3):
    # truth = CDF(Fv, Fbg[i], n1g[i], n2g[i])
    truth = dNdF(Fv, Ag[i], Fbg[i], n1g[i], n2g[i])
    true_hist = truth * Fv ** 2
    true_hist /= true_hist.sum()
    # truth_raw = CDF(Fv, Fbg[i], n1g[i], n2g[i])
    # truth = truth_raw * np.gradient(Fv) ** 0
    # truth /= truth.sum()
    # truth = truth.cumsum()
    plt.gcf().get_axes()[i].plot(Fv, true_hist.cumsum(), c='black', ls='--')

plt.gcf().savefig("benchmark_hist_cdf.pdf", bbox_inches="tight")

# # Integrate F2dNdF_true over logarithmically space bins
# F_bin_width = np.diff(F_eval)
# F2dNdF_running_int = np.array([np.abs(np.cumsum(0.5 * (F2dNdF_true[i][1:] + F2dNdF_true[i][:-1]) * np.log(F_bin_width))) for i in range(3)])
# # F2dNdF_true_cum = np.trapz(F2dNdF_true, np.log(gce.p.nn.hist["nn_hist_centers"]), axis=-1)
#
# for i in range(3):
#     # plt.gcf().get_axes()[i].semilogx(F_eval,  F2dNdF_true[i].cumsum() / (F2dNdF_true[i]).sum(), color="black", linestyle="--")
#     plt.gcf().get_axes()[i].semilogx(0.5 * (F_eval[1:] + F_eval[:-1]), F2dNdF_running_int[i] / F2dNdF_running_int[i][-1], color="black", linestyle="--")
# plt.gcf().savefig("benchmark_hist_cdf_normalized.pdf", bbox_inches="tight")