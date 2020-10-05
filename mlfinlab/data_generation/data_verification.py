"""
Contains methods for verifying synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from mlfinlab.codependence import get_dependence_matrix
from mlfinlab.clustering.hierarchical_clustering import optimal_hierarchical_cluster


def plot_time_series_dependencies(time_series, dependence_method="gnpr_distance", **kwargs):
    """
    Plots the dependence matrix of a time series returns.

    Used to verify a time series' underlying distributions via the GNPR distance method.
    ``**kwargs`` are used to pass arguments to the `get_dependence_matrix` function used here.

    :param time_series: (pd.DataFrame) Dataframe containing time series.
    :param dependence_method: (str) Distance method to use by `get_dependence_matrix`
    :return: (plt.Axes) Figure's axes.
    """

    return axis

def _compute_eigenvalues(mats):
    """
    Computes the eigenvalues of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param mats: (np.array) List of matrices to calculate eigenvalues from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting eigenvalues from mats.
    """

    pass


def _compute_pf_vec(mats):
    """
    Computes the Perron-Frobenius vector of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    The Perron-Frobenius property asserts that for a strictly positive square matrix, the
    corresponding eigenvector of the largest eigenvalue has strictly positive components.

    :param mats: (np.array) List of matrices to calculate Perron-Frobenius vector from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting Perron-Frobenius vectors from mats.
    """

    pass


def _compute_degree_counts(mats):
    """
    Computes the number of degrees in MST of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    The degree count is calculated by computing the MST of the matrix, and counting
    how many times each nodes appears in each edge produced by the MST. This count is normalized
    by the size of the matrix.

    :param mats: (np.array) List of matrices to calculate the number of degrees in MST from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting number of degrees in MST from mats.
    """

    pass


def plot_pairwise_dist(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Distribution of pairwise correlations is significantly shifted to the positive.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """

    pass


def plot_eigenvalues(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first eigenvalue (the market).

    - Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other large eigenvalues (industries).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """

    pass


def plot_eigenvectors(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Perron-Frobenius property (first eigenvector has positive entries).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """

    pass


def plot_hierarchical_structure(emp_mats, gen_mats):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Hierarchical structure of correlations.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :return: (tuple) Figures' axes.
    """

    pass


def plot_mst_degree_count(emp_mats, gen_mats):
    """
    Plots all the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Scale-free property of the corresponding Minimum Spanning Tree (MST).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :return: (plt.Axes) Figure's axes.
    """

    pass


def plot_stylized_facts(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    1. Distribution of pairwise correlations is significantly shifted to the positive.

    2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first
    eigenvalue (the market).

    3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other
    large eigenvalues (industries).

    4. Perron-Frobenius property (first eigenvector has positive entries).

    5. Hierarchical structure of correlations.

    6. Scale-free property of the corresponding Minimum Spanning Tree (MST).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    """

    pass


def plot_optimal_hierarchical_cluster(mat, method="ward"):
    """
    Calculates and plots the optimal clustering of a correlation matrix.

    It uses the `optimal_hierarchical_cluster` function in the clustering module to calculate
    the optimal hierarchy cluster matrix.

    :param mat: (np.array/pd.DataFrame) Correlation matrix.
    :param method: (str) Method to calculate the hierarchy clusters. Can take the values
        ["single", "complete", "average", "weighted", "centroid", "median", "ward"].
    :return: (plt.Axes) Figure's axes.
    """

    pass
