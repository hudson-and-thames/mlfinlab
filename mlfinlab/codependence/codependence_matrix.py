"""
This implementation lets user generate dependence and distance matrix based on the various methods of Information
Codependence  described in Cornell lecture notes on Codependence:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""

import numpy as np
import pandas as pd

from mlfinlab.codependence.information import variation_of_information_score, get_mutual_info
from mlfinlab.codependence.correlation import distance_correlation
from mlfinlab.codependence.gnpr_distance import spearmans_rho, gpr_distance, gnpr_distance
from mlfinlab.codependence.optimal_transport import optimal_transport_dependence


# pylint: disable=invalid-name

def get_dependence_matrix(df: pd.DataFrame, dependence_method: str, theta: float = 0.5,
                          n_bins: int = None, normalize: bool = True,
                          estimator: str = 'standard', target_dependence: str = 'comonotonicity',
                          gaussian_corr: float = 0.7, var_threshold: float = 0.2) -> pd.DataFrame:
    """
    This function returns a dependence matrix for elements given in the dataframe using the chosen dependence method.

    List of supported algorithms to use for generating the dependence matrix: ``information_variation``,
    ``mutual_information``, ``distance_correlation``, ``spearmans_rho``, ``gpr_distance``, ``gnpr_distance``,
    ``optimal_transport``.

    :param df: (pd.DataFrame) Features.
    :param dependence_method: (str) Algorithm to be use for generating dependence_matrix.
    :param theta: (float) Type of information being tested in the GPR and GNPR distances. Falls in range [0, 1].
                          (0.5 by default)
    :param n_bins: (int) Number of bins for discretization in ``information_variation`` and ``mutual_information``,
                         if None the optimal number will be calculated. (None by default)
    :param normalize: (bool) Flag used to normalize the result to [0, 1] in ``information_variation`` and
                             ``mutual_information``. (True by default)
    :param estimator: (str) Estimator to be used for calculation in ``mutual_information``.
                            [``standard``, ``standard_copula``, ``copula_entropy``] (``standard`` by default)
    :param target_dependence: (str) Type of target dependence to use in ``optimal_transport``.
                                    [``comonotonicity``, ``countermonotonicity``, ``gaussian``,
                                    ``positive_negative``, ``different_variations``, ``small_variations``]
                                    (``comonotonicity`` by default)
    :param gaussian_corr: (float) Correlation coefficient to use when creating ``gaussian`` and
                                  ``small_variations`` copulas. [from 0 to 1] (0.7 by default)
    :param var_threshold: (float) Variation threshold to use for coefficient to use in ``small_variations``.
                                  Sets the relative area of correlation in a copula. [from 0 to 1] (0.2 by default)
    :return: (pd.DataFrame) Dependence matrix.
    """

    pass


def get_distance_matrix(X: pd.DataFrame, distance_metric: str = 'angular') -> pd.DataFrame:
    """
    Applies distance operator to a dependence matrix.

    This allows to turn a correlation matrix into a distance matrix. Distances used are true metrics.

    List of supported distance metrics to use for generating the distance matrix: ``angular``, ``squared_angular``,
    and ``absolute_angular``.

    :param X: (pd.DataFrame) Dataframe to which distance operator to be applied.
    :param distance_metric: (str) The distance metric to be used for generating the distance matrix.
    :return: (pd.DataFrame) Distance matrix.
    """

    pass
