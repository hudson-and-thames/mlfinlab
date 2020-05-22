"""
This implementation lets user generate dependence and distance matrix based on the various methods of Information
Codependence  described in Cornell lecture notes: Codependence:
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
    This function returns a dependence matrix for the given method of dependence method.

    List of supported algorithms to use for generating the dependence_matrix: 'information_variation',
    'mutual_information', 'distance_correlation', 'spearmans_rho', 'gpr_distance', 'gnpr_distance'

    :param df: (pd.DataFrame) Features.
    :param dependence_method: (str) Algorithm to be use for generating dependence_matrix.
    :param theta: (float) Type of information being tested in the GPR and GNPR distances. Falls in range [0, 1]
    :param bandwidth: (float) Bandwidth to use for splitting observations in the GPR and GNPR distances.
    :return: (pd.DataFrame) Dependence_matrix.
    """
    # Get the feature names.
    features_cols = df.columns.values
    n = df.shape[1]
    np_df = df.values.T  # Make columnar access, but for np.array

    # Defining the dependence function.
    if dependence_method == 'information_variation':
        dep_function = variation_of_information_score
    elif dependence_method == 'mutual_information':
        dep_function = get_mutual_info
    elif dependence_method == 'distance_correlation':
        dep_function = distance_correlation
    elif dependence_method == 'spearmans_rho':
        dep_function = spearmans_rho
    elif dependence_method == 'gpr_distance':
        dep_function = gpr_distance
    elif dependence_method == 'gnpr_distance':
        dep_function = gnpr_distance
    else:
        raise ValueError(f"{dependence_method} is not a valid method. Use either 'information_variation'\
                                 or 'mutual_information' or 'distance_correlation'.")

    # TODO: rewtire in a better form
    # Generating the dependence_matrix for the defined method.
    if dependence_method in ('information_variation', 'mutual_information'):
        dependence_matrix = np.array([
            [
                dep_function(np_df[i], np_df[j], normalize=True) if j < i else
                # Leave diagonal elements as 0.5 to later double them to 1
                0.5 * dep_function(np_df[i], np_df[j], normalize=True) if j == i else
                0  # Make upper triangle 0 to fill it later on
                for j in range(n)
            ]
            for i in range(n)
        ])

    elif dependence_method == 'gpr_distance':
        dependence_matrix = np.array([
            [
                dep_function(np_df[i], np_df[j], theta=theta) if j < i else
                # Leave diagonal elements as 0.5 to later double them to 1
                0.5 * dep_function(np_df[i], np_df[j], theta=theta) if j == i else
                0  # Make upper triangle 0 to fill it later on
                for j in range(n)
            ]
            for i in range(n)
        ])

    elif dependence_method == 'gnpr_distance':
        dependence_matrix = np.array([
            [
                dep_function(np_df[i], np_df[j], theta=theta, bandwidth=bandwidth) if j < i else
                # Leave diagonal elements as 0.5 to later double them to 1
                0.5 * dep_function(np_df[i], np_df[j], theta=theta, bandwidth=bandwidth) if j == i else
                0  # Make upper triangle 0 to fill it later on
                for j in range(n)
            ]
            for i in range(n)
        ])

    else:
        dependence_matrix = np.array([
            [
                dep_function(np_df[i], np_df[j]) if j < i else
                0.5 * dep_function(np_df[i], np_df[j]) if j == i else  # Leave diagonal elements as 0.5 to later double them to 1
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
    Apply distance operator to a dependence matrix.

    :param X: (pd.DataFrame) Dataframe to which distance operator to be applied.
    :param distance_metric: (str) The distance operator to be used for generating the distance matrix.
       The methods that can be applied are: 'angular', 'squared_angular' and 'absolute_angular'.
    :return: (pd.DataFrame) Distance matrix
    """
    if distance_metric == 'angular':
        distfun = lambda x: ((1 - x).round(5) / 2.) ** .5
    elif distance_metric == 'abs_angular':
        distfun = lambda x: ((1 - abs(x)).round(5) / 2.) ** .5
    elif distance_metric == 'squared_angular':
        distfun = lambda x: ((1 - x ** 2).round(5) / 2.) ** .5
    else:
        raise ValueError(f'{distance_metric} is a unknown distance metric')

    return distfun(X).fillna(0)
