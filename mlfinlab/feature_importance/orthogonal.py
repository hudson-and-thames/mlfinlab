"""
Module which implements feature PCA compression and PCA analysis of feature importance.
"""

import pandas as pd
import numpy as np
from scipy.stats import weightedtau, kendalltau, spearmanr, pearsonr


def _get_eigen_vector(dot_matrix, variance_thresh, num_features=None):
    """
    Advances in Financial Machine Learning, Snippet 8.5, page 119.

    Computation of Orthogonal Features

    Gets eigen values and eigen vector from matrix which explain % variance_thresh of total variance.

    :param dot_matrix: (np.array): Matrix for which eigen values/vectors should be computed.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :param num_features: (int) Manually set number of features, overrides variance_thresh. (None by default)
    :return: (pd.Series, pd.DataFrame): Eigenvalues, Eigenvectors.
    """

    pass


def _standardize_df(data_frame):
    """
    Helper function which divides df by std and extracts mean.

    :param data_frame: (pd.DataFrame): Dataframe to standardize
    :return: (pd.DataFrame): Standardized dataframe
    """

    pass


def get_orthogonal_features(feature_df, variance_thresh=.95, num_features=None):
    """
    Advances in Financial Machine Learning, Snippet 8.5, page 119.

    Computation of Orthogonal Features.

    Gets PCA orthogonal features.

    :param feature_df: (pd.DataFrame): Dataframe of features.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :param num_features: (int) Manually set number of features, overrides variance_thresh. (None by default)
    :return: (pd.DataFrame): Compressed PCA features which explain %variance_thresh of variance.
    """

    pass


def get_pca_rank_weighted_kendall_tau(feature_imp, pca_rank):
    """
    Advances in Financial Machine Learning, Snippet 8.6, page 121.

    Computes Weighted Kendall's Tau Between Feature Importance and Inverse PCA Ranking.

    :param feature_imp: (np.array): Feature mean importance.
    :param pca_rank: (np.array): PCA based feature importance rank.
    :return: (float): Weighted Kendall Tau of feature importance and inverse PCA rank with p_value.
    """

    pass


def feature_pca_analysis(feature_df, feature_importance, variance_thresh=0.95):
    """
    Performs correlation analysis between feature importance (MDI for example, supervised) and PCA eigenvalues
    (unsupervised).

    High correlation means that probably the pattern identified by the ML algorithm is not entirely overfit.

    :param feature_df: (pd.DataFrame): Features dataframe.
    :param feature_importance: (pd.DataFrame): Individual MDI feature importance.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain in PCA compression.
    :return: (dict): Dictionary with kendall, spearman, pearson and weighted_kendall correlations and p_values.
    """

    pass
