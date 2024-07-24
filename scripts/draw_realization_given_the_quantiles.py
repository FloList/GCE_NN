import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skewnorm
from scipy.interpolate import interp1d

def generate_random_variable(n_samples, tau, quantiles):
    # Sort the quantiles in ascending order
    sort_idx = np.argsort(quantiles)
    quantiles = np.asarray(quantiles)[sort_idx]
    tau = np.array(tau)[sort_idx]
    # Add 0 and 1 to tau
    tau = np.concatenate([[0], tau, [1]])
    n_quantiles = len(tau)
    # Add min. and max. again to quantiles
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

loc = 3.0
scale = 0.2
skew = 3.0

tau = np.linspace(0.001, 0.999, 300)
quantiles = skewnorm.ppf(tau, loc=loc, scale=scale, a=skew)

n_samples = 10000
X = generate_random_variable(n_samples, tau, quantiles)

plt.plot(quantiles, skewnorm.pdf(quantiles, loc=loc, scale=scale, a=skew))
plt.hist(X, density=True, bins=50)
