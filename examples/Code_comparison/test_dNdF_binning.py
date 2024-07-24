import numpy as np
import matplotlib.pyplot as plt
from GCE.pdf_sampler import PDFSampler


# Nick's code
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

# Parameter
Ag = np.array([8.e10, 8.5e9, 3.1e12])
Fbg = np.array([2.e-11, 1.e-10, 1.e-12])
n1g = np.array([2.06, 2.8, 1.9])
n2g = np.array([-0.5, -0.1, -1.7])
N_tot = Ag * Fbg * (1 / (n1g - 1) + 1 / (1 - n2g))

bins_dndf = np.asarray([-np.infty] + list(np.logspace(-12.5, -7.0, 21)) + [np.infty])  # bins for SCD
power_of_f_dndf = 1.0
n_ps = 100000

fvals = np.logspace(-12.5, -9.0, 1000)
sampler = PDFSampler(fvals, dNdF(fvals, Ag[0], Fbg[0], n1g[0], n2g[0]))
samples = sampler(n_ps)

# Plot
true_integral = np.trapz(dNdF(fvals, Ag[0], Fbg[0], n1g[0], n2g[0]) / N_tot[0], fvals)
true_pdf = dNdF(fvals, Ag[0], Fbg[0], n1g[0], n2g[0]) / N_tot[0]

true_pdf_at_bin_centers = dNdF(10 ** (np.log10(bins_dndf[:-1]) + np.diff(np.log10(bins_dndf)) / 2), Ag[0], Fbg[0], n1g[0], n2g[0]) / N_tot[0]
true_integral_at_bin_centers = np.trapz(true_pdf_at_bin_centers, 10 ** (np.log10(bins_dndf[:-1]) + np.diff(np.log10(bins_dndf)) / 2))


plt.figure()
plt.hist(samples, bins=bins_dndf, histtype='step', weights=np.ones_like(samples), label="Histogram")
plt.plot(fvals, true_pdf, label="True dNdF")
plt.xscale("log")
plt.yscale("log")


flux_hist, bin_edges = np.histogram(samples, weights=samples ** power_of_f_dndf, bins=bins_dndf)

dlogF = np.diff(np.log10(bin_edges))[1]
bin_centers = 10 ** (np.log10(bin_edges[:-1]) + dlogF / 2)
bin_centers[0] = 10 ** (np.log10(bin_edges[1]) - dlogF / 2)
bin_centers[-1] = 10 ** (np.log10(bin_edges[-2]) + dlogF / 2)

# plt.figure()
# plt.loglog(bin_centers, flux_hist)

# Restore the dNdF from flux_hist
bin_edges_corr = np.copy(bin_edges)
bin_edges[0] = 10 ** (np.log10(bin_edges[1]) - dlogF)
bin_edges[-1] = 10 ** (np.log10(bin_edges[-2]) + dlogF)
dNdF_restored_raw = flux_hist / (np.diff(bin_edges) * N_tot[0])
dNdF_restored_integral = np.trapz(dNdF_restored_raw, bin_centers)
dNdF_restored = dNdF_restored_raw / dNdF_restored_integral

plt.plot(bin_centers, dNdF_restored, label="Restored")
plt.legend()

# Also plot F^2 dN/dF
plt.figure()
plt.loglog(bin_centers, bin_centers ** 2 * dNdF_restored)
plt.plot(fvals, fvals ** 2 * dNdF(fvals, Ag[0], Fbg[0], n1g[0], n2g[0]) / N_tot[0], label="dNdF")
