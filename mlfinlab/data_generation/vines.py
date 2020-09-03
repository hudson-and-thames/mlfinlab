# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Implementation of generating financial correlation matrices from
"Generating random correlation matrices based on vines and extended onion method"
by Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
https://www.sciencedirect.com/science/article/pii/S0047259X09000876
and "Generating random correlation matrices based partial correlations" by Harry Joe.
https://www.sciencedirect.com/science/article/pii/S0047259X05000886
"""
import numpy as np


def _correlation_from_partial_dvine(partial_correlations, a_beta, b_beta, row, col):
    """
    Calculates a correlation based on partical correlations using the D-vine method.

    It samples from a beta distribution, adjusts it to the range [-1, 1]. Sets this value
    as the starting partial correlation, and follows the D-vine to calculate the final
    correlation.

    :param partial_correlations: (np.array) Matrix of current partial correlations. It is
        modified during this function's execution.
    :param a_beta: (float) Alpha parameter of the beta distribution to sample from.
    :param b_beta: (float) Beta parameter of the beta distribution to sample from.
    :param row: (int) Starting row of the partial correlation matrix.
    :param col: (int) Starting column of the partial correlation matrix.
    :return: (float) Calculated correlation.
    """

    pass


def _correlation_from_partial_cvine(partial_correlations, a_beta, b_beta, row, col):
    """
    Calculates a correlation based on partical correlations using the C-vine method.

    It samples from a beta distribution, adjusts it to the range [-1, 1]. Sets this value
    as the starting partial correlation, and follows the C-vine to calculate the final
    correlation.

    :param partial_correlations: (np.array) Matrix of current partial correlations. It is
        modified during this function's execution.
    :param a_beta: (float) Alpha parameter of the beta distribution to sample from.
    :param b_beta: (float) Beta parameter of the beta distribution to sample from.
    :param row: (int) Starting row of the partial correlation matrix.
    :param col: (int) Starting column of the partial correlation matrix.
    :return: (float) Calculated correlation.
    """

    pass


def _q_vector_correlations(corr_mat, r_factor, dim):
    """
    Sample from unit vector uniformly on the surface of the k_loc-dimensional hypersphere and
    obtains the q vector of correlations.

    :param corr_mat (np.array) Correlation matrix.
    :param r_factor (np.array) R factor vector based on correlation matrix.
    :param dim: (int) Dimension of the hypersphere to sample from.
    :return: (np.array) Q vector of correlations.
    """

    pass


def sample_from_dvine(dim=10, n_samples=1, beta_dist_fixed=None):
    """
    Generates uniform correlation matrices using the D-vine method.

    It is reproduced with modifications from the following paper:
    `Joe, H., 2006. Generating random correlation matrices based on partial correlations.
    Journal of Multivariate Analysis, 97(10), pp.2177-2189.
    <https://www.sciencedirect.com/science/article/pii/S0047259X05000886>`_

    It uses the partial correlation D-vine to generate partial correlations. The partial
    correlations
    are sampled from a uniform beta distribution and adjusted to thr range [-1, 1]. Then these
    partial correlations are converted into raw correlations by using a recursive formula based
    on its location on the vine.

    :param dim: (int) Dimension of correlation matrix to generate.
    :param n_samples: (int) Number of samples to generate.
    :param beta_dist_fixed: (tuple) Overrides the beta distribution parameters. The input is
        two float parameters (alpha, beta), used in the distribution. (None by default)
    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).
    """

    pass


def sample_from_cvine(dim=10, eta=2, n_samples=1, beta_dist_fixed=None):
    """
    Generates uniform correlation matrices using the C-vine method.

    It is reproduced with modifications from the following paper:
    `Lewandowski, D., Kurowicka, D. and Joe, H., 2009. Generating random correlation matrices based
    on vines and extended onion method. Journal of multivariate analysis, 100(9), pp.1989-2001.
    <https://www.sciencedirect.com/science/article/pii/S0047259X09000876>`_

    It uses the partial correlation C-vine to generate partial correlations. The partial
    correlations
    are sampled from a uniform beta distribution proportional to its determinant and the factor
    eta.
    and adjusted to thr range [-1, 1]. Then these partial correlations are converted into raw
    correlations by using a recursive formula based on its location on the vine.

    :param dim: (int) Dimension of correlation matrix to generate.
    :param eta: (int) Corresponds to uniform distribution of beta.
        Correlation matrix `S` has a distribution proportional to [det C]^(eta - 1)
    :param n_samples: (int) Number of samples to generate.
    :param beta_dist_fixed: (tuple) Overrides the beta distribution parameters. The input is
        two float parameters (alpha, beta), used in the distribution. (None by default)
    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).
    """

    pass


def sample_from_ext_onion(dim=10, eta=2, n_samples=1):
    """
    Generates uniform correlation matrices using extended onion method.

    It is reproduced with modifications from the following paper:
    `Lewandowski, D., Kurowicka, D. and Joe, H., 2009. Generating random correlation matrices based
    on vines and extended onion method. Journal of multivariate analysis, 100(9), pp.1989-2001.
    <https://www.sciencedirect.com/science/article/pii/S0047259X09000876>`_

    It uses the extended onion to generate correlations sampled from a uniform beta distribution.
    It starts with a one-dimensional matrix, and it iteratively grows the matrix by adding extra
    rows and columns by sampling from the convex, closed, compact and full-dimensional set on the
    surface of a k-dimensional hypersphere.

    :param dim: (int) Dimension of correlation matrix to generate.
    :param eta: (int) Corresponds to uniform distribution of beta.
        Correlation matrix `S` has a distribution proportional to [det C]^(eta - 1)
    :param n_samples: (int) Number of samples to generate.
    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).
    """

    pass
