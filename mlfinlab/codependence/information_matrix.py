"""
This script implements the creation of dependence and distance matrix for the
methods of Information Codependence.
"""

#Imports
import numpy as np
import pandas as pd

from mlfinlab.codependence.information import variation_of_information_score,mutual_info_score
from mlfinlab.codependence.correlation import angular_distance,absolute_angular_distance,\
                                              squared_angular_distance,distance_correlation


def get_dependence_matrix(df : pd.DataFrame, base_algorithm : str ,n_bins : int = None, normalize : bool = True):
    """
    This function returns dependence and distance matrix for the methods of Information Codependence.

    :param df: (pd.DataFrame) of features
    :param base_algorithm: (str) the algorithm to use to get the dependence_matrix,
                                either 'variation_of_information' or 'mutual_information' .
    :param n_bins: (int) number of bins for discretization, if None get number of bins based on correlation coefficient.
    :param normalize: (bool) True to normalize the result to [0, 1].
    :return: (pd.DataFrame)  of dependence_matrix
    """

    info_mtx = pd.DataFrame(index=df.columns)
    features_cols = df.columns.values

    algo = None
    if base_algorithm == 'variation_of_information':
        algo = variation_of_information_score
    elif base_algorithm == 'mutual_information':
        algo = mutual_info_score
    else:
        raise ValueError(f"{base_algorithm} is not a base algorithm. Use either 'variation_of_information' \
                            or 'mutual_information' ")
    for col0 in features_cols:
        vi = []
        for col1 in features_cols:
            x = df[col0].values
            y = df[col1].values
            vi.append(algo(x,y,n_bins,normalize))
        info_mtx[col0] = vi
    return info_mtx

def get_distance_matrix(X : pd.DataFrame or np.array, transformation : str = 'angular'):

    """
    This function returns the distance_matrix for the given metrix X.

    :param X: (pd.DataFrame or np.array) of which distance is to calculated
    :param transformation: (str) the transformation method to used for getting the distance matrix.
                                The transformation methods that can be applied are -
                                'angular' , 'squared_angular' and 'absolute_angular'.
    :return: (pd.DataFrame or np.array) of distance matrix
    """

    if transformation == 'angular':
        trns_mtx = np.sqrt((1 - X).round(5)/2)
    elif transformation == 'absolute_angular':
        trns_mtx = np.sqrt((1 - abs(X)).round(5)/2)
    elif transformation == 'squared_angular':
        trns_mtx = np.sqrt((1 - X ** 2).round(5)/2)
    else:
        raise ValueError(f"{transformation} is not a vaild transformation method.")
    return trns_mtx
