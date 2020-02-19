"""
Module which implements feature PCA compression and PCA analysis of feature importance.
"""

import pandas as pd
import numpy as np
from scipy.stats import weightedtau, kendalltau, spearmanr, pearsonr


def _get_eigen_vector(dot_matrix, variance_thresh):
    """
    Snippet 8.5, page 119. Computation of Orthogonal Features

    Get eigen values and eigen vector from matrix which explain % variance_thresh of total variance.

    :param dot_matrix: (np.array): Matrix for which eigen values/vectors should be computed.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :return: (pd.Series, pd.DataFrame): Eigen values, Eigen vectors.
    """
    # Compute eigen_vec from dot prod matrix, reduce dimension
    eigen_val, eigen_vec = np.linalg.eigh(dot_matrix)
    idx = eigen_val.argsort()[::-1]  # Arguments for sorting eigen_val desc
    eigen_val, eigen_vec = eigen_val[idx], eigen_vec[:, idx]

    # 2) Only positive eigen_vals
    eigen_val = pd.Series(eigen_val, index=['PC_' + str(i + 1) for i in range(eigen_val.shape[0])])
    eigen_vec = pd.DataFrame(eigen_vec, index=dot_matrix.index, columns=eigen_val.index)
    eigen_vec = eigen_vec.loc[:, eigen_val.index]

    # 3) Reduce dimension, form PCs
    cum_var = eigen_val.cumsum() / eigen_val.sum()
    dim = cum_var.values.searchsorted(variance_thresh)
    eigen_val, eigen_vec = eigen_val.iloc[:dim + 1], eigen_vec.iloc[:, :dim + 1]
    return eigen_val, eigen_vec


def _standardize_df(data_frame):
    """
    Helper function which divides df by std and extracts mean.

    :param data_frame: (pd.DataFrame): to standardize
    :return: (pd.DataFrame): standardized data frame
    """
    return data_frame.sub(data_frame.mean(), axis=1).div(data_frame.std(), axis=1)


def get_orthogonal_features(feature_df, variance_thresh=.95):
    """
    Snippet 8.5, page 119. Computation of Orthogonal Features.

    Get PCA orthogonal features.

    :param feature_df: (pd.DataFrame): Dataframe of features.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :return: (pd.DataFrame): Compressed PCA features which explain %variance_thresh of variance.
    """
    # Given a dataframe of features, compute orthogonal features
    feature_df_standard = _standardize_df(feature_df)  # Standardize
    dot_matrix = pd.DataFrame(np.dot(feature_df_standard.T, feature_df_standard), index=feature_df.columns,
                              columns=feature_df.columns)
    _, eigen_vec = _get_eigen_vector(dot_matrix, variance_thresh)
    pca_features = np.dot(feature_df_standard, eigen_vec)
    return pca_features


def get_pca_rank_weighted_kendall_tau(feature_imp, pca_rank):
    """
    Snippet 8.6, page 121. Computation of Weighted Kendall's Tau Between Feature Importance and Inverse PCA Ranking.

    :param feature_imp: (np.array): Feature mean importance.
    :param pca_rank: (np.array): PCA based feature importance rank.
    :return: (float): Weighted Kendall Tau of feature importance and inverse PCA rank with p_value.
    """
    return weightedtau(feature_imp, pca_rank ** -1.0)


def feature_pca_analysis(feature_df, feature_importance, variance_thresh=0.95):
    """
    Perform correlation analysis between feature importance (MDI for example, supervised) and PCA eigen values
    (unsupervised). High correlation means that probably the pattern identified by the ML algorithm is not entirely overfit.

    :param feature_df: (pd.DataFrame): Features dataframe.
    :param feature_importance: (pd.DataFrame): Individual MDI feature importance.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain in PCA compression.
    :return: (dict): Dictionary with kendall, spearman, pearson and weighted_kendall correlations and p_values.
    """
    feature_df_standard = _standardize_df(feature_df)  # Standardize
    dot = pd.DataFrame(np.dot(feature_df_standard.T, feature_df_standard), index=feature_df.columns,
                       columns=feature_df.columns)
    eigen_val, eigen_vec = _get_eigen_vector(dot, variance_thresh)

    # Compute correlations between eigen values for each eigen vector vs mdi importance
    all_eigen_values = []  # All eigen values in eigen vectors
    corr_dict = {'Pearson': [], 'Spearman': [], 'Kendall': []}  # Dictionary containing correlation metrics
    for vec in eigen_vec.columns:
        all_eigen_values.extend(abs(eigen_vec[vec].values * eigen_val[vec]))

    # We need to repeat importance array # of eigen vector times to generate correlation for all_eigen_values
    repeated_importance_array = np.tile(feature_importance['mean'].values, len(eigen_vec.columns))

    for corr_type, function in zip(corr_dict.keys(), [pearsonr, spearmanr, kendalltau]):
        corr_coef = function(repeated_importance_array, all_eigen_values)
        corr_dict[corr_type] = corr_coef

    # Get Rank based weighted Tau correlation
    feature_pca_rank = (eigen_val * eigen_vec).abs().sum(axis=1).rank(
        ascending=False)  # Sum of absolute values across all eigen vectors
    corr_dict['Weighted_Kendall_Rank'] = get_pca_rank_weighted_kendall_tau(feature_importance['mean'].values,
                                                                           feature_pca_rank.values)
    return corr_dict
