import numpy as np
from scipy.stats import norm


def expected_improvement(mean, var, y_min, ei_xi=0.0, *args, **kwargs):
    mean, var = np.atleast_1d(mean, var)
    if len(mean) != len(var):
        raise ValueError("len(mean) != len(var)")

    ei = np.zeros(shape=var.shape)
    ei[var <= 1e-10] = 0
    std = var[var > 1e-10]**0.5
    Z = (y_min-mean[var > 1e-10]-ei_xi) / std
    ei[var > 1e-10] = std * (Z * norm.cdf(Z) + norm.pdf(Z))

    return ei


def upper_confidence_bound(mean, var, ucb_kappa=2.0, *args, **kwargs):
    mean, var = np.atleast_1d(mean, var)
    if len(mean) != len(var):
        raise ValueError("len(mean) != len(var)")

    ucb = np.copy(mean)
    std = var[var > 1e-10] ** 0.5
    ucb[var > 1e-10] += ucb_kappa * std

    return ucb


acquisition_map = {"ei": expected_improvement,
                   "ucb": upper_confidence_bound}

