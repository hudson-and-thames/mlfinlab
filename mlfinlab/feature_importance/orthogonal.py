import pandas as pd
import numpy as np
from scipy.stats import weightedtau


def _get_eigen_vector(dot_matrix, variance_thresh):
    """
    Snippet 8.5, page 119. Computation of Orthogonal Features

    Get eigen values and eigen vector from matrix which explain % variance_thresh of total variance
    :param dot_matrix: (np.array): matrix for which eigen values/vectors should be computed
    :param variance_thresh: (float): % of overall variance which compressed vectors should explain
    :return: (pd.Series, pd.DataFrame): eigen values, eigen vectors
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


def _standardize_df(df):
    """
    Helper function which divides df by std and extracts mean.

    :param df: (pd.DataFrame): to standardize
    :return: (pd.DataFrame): standardized data frame
    """
    return df.sub(df.mean(), axis=1).div(df.std(), axis=1)


def get_orthogonal_features(feature_df, variance_thresh=.95):
    """
    Snippet 8.5, page 119. Computation of Orthogonal Features.

    Get PCA orthogonal features
    :param feature_df: (pd.DataFrame): with features
    :param variance_thresh: (float): % of overall variance which compressed vectors should explain
    :return: (pd.DataFrame): compressed PCA features which explain %variance_thresh of variance
    """
    # Given a dataframe  of features, compute orthofeatures 
    feature_df_standard = _standardize_df(feature_df)  # Standardize
    dot_matrix = pd.DataFrame(np.dot(feature_df_standard.T, feature_df_standard), index=feature_df.columns,
                              columns=feature_df.columns)
    eigen_val, eigen_vec = _get_eigen_vector(dot_matrix, variance_thresh)
    pca_features = np.dot(feature_df_standard, eigen_vec)
    return pca_features


def get_weighted_kendall_tau(feature_imp, pca_rank):
    return weightedtau(feature_imp, pca_rank ** -1.)[0]


def get_pca_analysis(feature_df, feature_importance, variance_thresh=0.95):
    """
    Perform correlation analysis between feature importance (MDI for example, supervised)
    and PCA eigen values (unsupervised). High correlation means that probably the pattern identified
    by the ML algorithm is not entirely overfit.

    :param feature_df: (pd.DataFrame): with features
    :param feature_importance: (pd.DataFrame): individual MDI feature importance
    :param variance_thresh: (float): % of overall variance which compressed vectors should explain in PCA compression
    :return: (dict): with kendall, spearman, pearson and weighted_kendall correlations
    """
    feature_df_standard = _standardize_df(feature_df)  # Standardize
    dot = pd.DataFrame(np.dot(feature_df_standard.T, feature_df_standard), index=feature_df.columns,
                       columns=feature_df.columns)
    eigen_val, eigen_vec = _get_eigen_vector(dot, variance_thresh)
    pass
