"""
Contains methods for generating correlated random walks.
"""

import numpy as np
import pandas as pd


def generate_cluster_time_series(n_series, t_samples=100, k_corr_clusters=1,
                                 d_dist_clusters=1, rho_main=0.1, rho_corr=0.3, price_start=100.0,
                                 dists_clusters=("normal", "normal", "student-t", "normal", "student-t")):
    """
    Generates a synthetic time series of correlation and distribution clusters.

    It is reproduced with modifications from the following paper:
    `Donnat, P., Marti, G. and Very, P., 2016. Toward a generic representation of random
    variables for machine learning. Pattern Recognition Letters, 70, pp.24-31.
    <https://www.sciencedirect.com/science/article/pii/S0167865515003906>`_

    `www.datagrapple.com. (n.d.). DataGrapple - Tech: A GNPR tutorial: How to cluster random walks.
    [online] Available at:  [Accessed 26 Aug. 2020].
    <https://www.datagrapple.com/Tech/GNPR-tutorial-How-to-cluster-random-walks.html>`_

    This method creates `n_series` time series of length `t_samples`. Each time series is divided
    into `k_corr_clusters` correlation clusters. Each correlation cluster is subdivided into
    `d_dist_clusters` distribution clusters.
    A main distribution is sampled from a normal distribution with mean = 0 and stdev = 1, adjusted
    by a `rho_main` factor. The correlation clusters are sampled from a given distribution, are generated
    once, and adjusted by a `rho_corr` factor. The distribution clusters are sampled from other
    given distributions, and adjusted by (1 - `rho_main` - `rho_corr`). They are sampled for each time series.
    These three series are added together to form a time series of returns. The final time series
    is the cumulative sum of the returns, with a start price given by `price_start`.

    :param n_series: (int) Number of time series to generate.
    :param t_samples: (int) Number of samples in each time series.
    :param k_corr_clusters: (int) Number of correlation clusters in each time series.
    :param d_dist_clusters: (int) Number of distribution clusters in each time series.
    :param rho_main: (float): Strength of main time series distribution.
    :param rho_corr: (float): Strength of correlation cluster distribution.
    :param price_start: (float) Starting price of the time series.
    :param dists_clusters: (list) List containing the names of the distributions to sample from.
        The following numpy distributions are available: "normal" = normal(0, 1), "normal_2" = normal(0, 2),
        "student-t" = standard_t(3)/sqrt(3), "laplace" = laplace(1/sqrt(2)). The first disitribution
        is used to sample for the correlation clusters (k_corr_clusters), the remaining ones are used
        to sample for the distribution clusters (d_dist_clusters).
    :return: (pd.DataFrame) Generated time series. Has size (t_samples, n_series).
    """

    pass
