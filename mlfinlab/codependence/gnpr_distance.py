"""
Implementation of distance using the Generic Non-Parametric Representation approach from "Some contributions to the
clustering of financial time series and applications to credit default swaps" by Gautier Marti
https://www.researchgate.net/publication/322714557
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import ot

# pylint: disable=invalid-name


def spearmans_rho(x: np.array, y: np.array) -> float:
    """
    Calculates a statistical estimate of Spearman's rho - a copula-based dependence measure.

    Formula for calculation:
    rho = 1 - (6)/(T*(T^2-1)) * Sum((X_t-Y_t)^2)

    It is more robust to noise and can be defined if the variables have an infinite second moment.
    This statistic is described in more detail in the work by Gautier Marti
    https://www.researchgate.net/publication/322714557 (p.54)

    This method is a wrapper for the scipy spearmanr function. For more details about the function and its parameters,
    please visit scipy documentation
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html

    :param x: (np.array/pd.Series) X vector
    :param y: (np.array/pd.Series) Y vector (same number of observations as X)
    :return: (float) Spearman's rho statistical estimate
    """

    # Coefficient calculationS

    pass


def gpr_distance(x: np.array, y: np.array, theta: float) -> float:
    """
    Calculates the distance between two Gaussians under the Generic Parametric Representation (GPR) approach.

    According to the original work https://www.researchgate.net/publication/322714557 (p.70):
    "This is a fast and good proxy for distance d_theta when the first two moments ... predominate". But it's not
    a good metric for heavy-tailed distributions.

    Parameter theta defines what type of information dependency is being tested:
    - for theta = 0 the distribution information is tested
    - for theta = 1 the dependence information is tested
    - for theta = 0.5 a mix of both information types is tested

    With theta in [0, 1] the distance lies in range [0, 1] and is a metric. (See original work for proof, p.71)

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector (same number of observations as X).
    :param theta: (float) Type of information being tested. Falls in range [0, 1].
    :return: (float) Distance under GPR approach.
    """

    pass


def gnpr_distance(x: np.array, y: np.array, theta: float, n_bins: int = 50) -> float:
    """
    Calculates the empirical distance between two random variables under the Generic Non-Parametric Representation
    (GNPR) approach.

    Formula for the distance is taken from https://www.researchgate.net/publication/322714557 (p.72).

    Parameter theta defines what type of information dependency is being tested:
    - for theta = 0 the distribution information is tested
    - for theta = 1 the dependence information is tested
    - for theta = 0.5 a mix of both information types is tested

    With theta in [0, 1] the distance lies in the range [0, 1] and is a metric.
    (See original work for proof, p.71)

    This method is modified as it uses 1D Optimal Transport Distance to measure
    distribution distance. This solves the issue of defining support and choosing
    a number of bins. The number of bins can be given as an input to speed up calculations.
    Big numbers of bins can take a long time to calculate.

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector (same number of observations as X).
    :param theta: (float) Type of information being tested. Falls in range [0, 1].
    :param n_bins: (int) Number of bins to use to split the X and Y vector observations.
        (100 by default)
    :return: (float) Distance under GNPR approach.
    """

    pass
