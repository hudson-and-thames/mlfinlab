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

# pylint: disable=invalid-name

def get_dependence_matrix(df: pd.DataFrame, dependence_method: str, theta: float = 0.5,
                          bandwidth: float = 0.01) -> pd.DataFrame:
    """
    This function returns a dependence matrix for elements given in the dataframe using the chosen dependence method.

    List of supported algorithms to use for generating the dependence matrix: ``information_variation``,
    ``mutual_information``, ``distance_correlation``, ``spearmans_rho``, ``gpr_distance``, ``gnpr_distance``.

    :param df: (pd.DataFrame) Features.
    :param dependence_method: (str) Algorithm to be use for generating dependence_matrix.
    :param theta: (float) Type of information being tested in the GPR and GNPR distances. Falls in range [0, 1].
                          (0.5 by default)
    :param bandwidth: (float) Bandwidth to use for splitting observations in the GPR and GNPR distances. (0.01 by default)
    :return: (pd.DataFrame) Dependence matrix.
    """
    # Get the feature names.
    features_cols = df.columns.values
    n = df.shape[1]
    np_df = df.values.T  # Make columnar access, but for np.array

    # Defining the dependence function.
    if dependence_method == 'information_variation':
        dep_function = lambda x, y: variation_of_information_score(x, y, normalize=True)
    elif dependence_method == 'mutual_information':
        dep_function = lambda x, y: get_mutual_info(x, y, normalize=True)
    elif dependence_method == 'distance_correlation':
        dep_function = distance_correlation
    elif dependence_method == 'spearmans_rho':
        dep_function = spearmans_rho
    elif dependence_method == 'gpr_distance':
        dep_function = lambda x, y: gpr_distance(x, y, theta=theta)
    elif dependence_method == 'gnpr_distance':
        dep_function = lambda x, y: gnpr_distance(x, y, theta=theta, bandwidth=bandwidth)
    else:
        raise ValueError(f"{dependence_method} is not a valid method. Please use one of the supported methods \
                            listed in the docsting.")

    # Generating the dependence_matrix
    dependence_matrix = np.array([
        [
            dep_function(np_df[i], np_df[j]) if j < i else
            # Leave diagonal elements as 0.5 to later double them to 1
            0.5 * dep_function(np_df[i], np_df[j]) if j == i else
            0  # Make upper triangle 0 to fill it later on
            for j in range(n)
        ]
        for i in range(n)
    ])

    # Make matrix symmetrical
    dependence_matrix = dependence_matrix + dependence_matrix.T

    #  Dependence_matrix converted into a DataFrame.
    dependence_df = pd.DataFrame(data=dependence_matrix, index=features_cols, columns=features_cols)

    if dependence_method == 'information_variation':
        return 1 - dependence_df  # IV is reverse, 1 - independent, 0 - similar

    return dependence_df

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
    if distance_metric == 'angular':
        distfun = lambda x: ((1 - x).round(5) / 2.) ** .5
    elif distance_metric == 'abs_angular':
        distfun = lambda x: ((1 - abs(x)).round(5) / 2.) ** .5
    elif distance_metric == 'squared_angular':
        distfun = lambda x: ((1 - x ** 2).round(5) / 2.) ** .5
    else:
        raise ValueError(f'{distance_metric} is a unknown distance metric. Please use one of the supported methods \
                            listed in the docsting.')

    return distfun(X).fillna(0)
