"""
Implementation of distance using the Generic Non-Parametric Representation approach from "Some contributions to the
clustering of financial time series and applications to credit default swaps" by Gautier Marti
https://www.researchgate.net/publication/322714557
"""
import numpy as np

# pylint: disable=invalid-name

def spearmans_rho(x: np.array, y: np.array) -> float:
    """
    Calculates a statistical estimate of Spearman's rho - a copula-based dependence measure.

    It is more robust to noise and can be defined if the variables have an infinite second moment.

    Formula for the statistic is taken form https://www.researchgate.net/publication/322714557 (p.54)

    rho = 1 - (6)/(T*(T^2-1)) * Sum((X_t-Y_t)^2)

    :param x: (np.array) X vector
    :param y: (np.array) Y vector (same number of observations as X)
    :return: (float) Spearman's rho statistical estimate
    """

    # Number of observations
    num_obs = x.shape[0]

    # Coefficient calculation
    rho = 1 - (6) / (num_obs * (num_obs**2 - 1)) * (np.power(x - y, 2).sum())

    return rho


def gnp_distance(x: np.array, y: np.array, theta: float) -> float:
    """
    Calculates the distance between to Gaussians under the Generic Parametric Representation (GPR) approach.

    According to the original work https://www.researchgate.net/publication/322714557 (p.70):
    "This is a fast and good proxy for distance d_theta when the first two moments ... predominate". But it's not
    a good metric for heavy-tailed distributions.

    Parameter theta defines what type of information dependency is being tested:
    - for theta = 0 the distribution information is tested
    - for theta = 1 the dependence information is tested
    - for theta = 0.5 a mix of both information types is tested

    With theta in [0, 1] the distance lies in range [0, 1] and is a metric. (See original work for proof, p.71)

    :param x: (np.array) X vector
    :param y: (np.array) Y vector (same number of observations as X)
    :param theta: (float) type of information being tested. Falls in range [0, 1]
    :return: (float) Distance under GPR approach
    """

    # Calculating the distance
    distance = theta * (1 - spearmans_rho(x, y)) / 2 + \
               (1 - theta) * (1 - ((2 * x.std() * y.std()) /(x.std()**2 + y.std()**2)) *
                              np.exp(- (x.mean() - y.mean())**2 / (x.std()**2 + y.std()**2)))

    return distance**(1/2)
