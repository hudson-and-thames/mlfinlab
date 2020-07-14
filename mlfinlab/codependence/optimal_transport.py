"""
Implementations of Optimal Transport dependence measure proposed by Marti et al. : https://arxiv.org/abs/1610.09659
And implemented in the blog post by Marti: https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html
"""
import numpy as np
import scipy.stats as ss
import ot


# pylint: disable=invalid-name

def _get_empirical_copula(x: np.array, y: np.array) -> np.array:
    """
    Calculate empirical copula using ranked observations.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :return: (np.array) Empirical copula.
    """

    # Calculate ranked observations
    x_unif = ss.rankdata(x) / len(x)
    y_unif = ss.rankdata(y) / len(y)

    # Empirical copula
    empirical = np.array([[x, y] for x, y in zip(x_unif, y_unif)])

    return empirical

def optimal_transport_distance(x: np.array, y: np.array, target_dependence: str = 'comonotonicity',
                               gaussian_corr: float = 0.7, var_threshold: float = 0.2) -> float:
    """
    Calculates optimal transport distance between two vectors.

    This implementation is based on the blog post by Marti:
    https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html

    The target and forget copulas are being used to reference where between them does the empirical
    copula stand in the space of copulas. The forget copula used is the copula associated with
    independent random variables. The target copula is defined by the target_dependence parameter.

    Currently, these target_dependence copulas are supported:

    - ``comonotonicity`` - a comonotone copula.
    - ``countermonotonicity`` - a countermonotone copula.
    - ``gaussian`` - a Gaussian copula with a custom correlation coefficient.
    - ``positive_negative`` - a copula of both positive and negative correlation.
    - ``different_variations`` - a copula with some elements having extreme variations,
      while those of others are relatively small, and conversely.
    - ``small_variations`` - a copula with elements being positively correlated for small variations
      but uncorrelated otherwise.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param target_dependence: (str) Type of target dependence to use when measuring distance.
                                    (``comonotonicity`` by default)
    :param gaussian_corr: (float) Correlation coefficient to use when creating ``gaussian`` and
                                  ``small_variations`` copulas. [from 0 to 1] (0.7 by default)
    :param var_threshold: (float) Variation threshold to use for coefficient to use in ``small_variations``.
                                  Sets the relative area of correlation in a copula. [from 0 to 1] (0.2 by default)
    :return: (float) Optimal transport distance.
    """

    # Defining a number of observations used
    n_obs = x.shape[0]

    # Creating forget copula with independence
    forget = np.array([[u, v] for u, v in zip(np.random.uniform(size=n_obs),
                                              np.random.uniform(size=n_obs))])

    # Creating target copula with a given dependence type
    target = _create_target_copula(target_dependence, n_obs, gaussian_corr, var_threshold)

    # Creating empirical copula from observations
    empirical = _get_empirical_copula(x, y)

    # Optimal transport distance
    copula_ot = _compute_copula_ot_dependence(empirical, target, forget, n_obs)

    return copula_ot

def _compute_copula_ot_dependence(empirical: np.array, target: np.array, forget: np.array, n_obs: int) -> float:
    """
    Calculates optimal copula transport dependence measure.

    :param empirical: (np.array) Empirical copula.
    :param target: (np.array) Target copula.
    :param forget: (np.array) Forget copula.
    :param nb_obs: (int) Number of observations.
    :return: (float) Optimal transport dependence.
    """

    # Uniform distribution on samples
    t_measure, f_measure, e_measure = (np.ones((n_obs,)) / n_obs,
                                       np.ones((n_obs,)) / n_obs,
                                       np.ones((n_obs,)) / n_obs)

    # Compute the ground distance matrix between locations
    gdist_e2t = ot.dist(empirical, target)
    gdist_e2f = ot.dist(empirical, forget)

    # Compute the optimal transport matrix
    e2t_ot = ot.emd(t_measure, e_measure, gdist_e2t)
    e2f_ot = ot.emd(f_measure, e_measure, gdist_e2f)

    # Compute the optimal transport distance:
    # <optimal transport matrix, ground distance matrix>_F
    e2t_dist = np.trace(np.dot(np.transpose(e2t_ot), gdist_e2t))
    e2f_dist = np.trace(np.dot(np.transpose(e2f_ot), gdist_e2f))

    # Compute the copula ot dependence measure
    ot_measure = 1 - e2t_dist / (e2f_dist + e2t_dist)

    return ot_measure

def _create_target_copula(target_dependence: str, n_obs: int, gauss_corr: float, var_threshold: float) -> np.array:
    """
    Creates target copula with given dependence and number of observations.

    :param target_dependence: (str) Type of dependence to use for copula creation.[``comonotonicity``,
                                    ``countermonotonicity``, ``gaussian``, ``positive_negative``,
                                    ``different_variations``, ``small_variations``]
    :param n_obs: (int) Number of observations to use for copula creation.
    :param gauss_corr: (float) Correlation coefficient to use when creating ``gaussian`` and
                                  ``small_variations`` copulas.
    :param var_threshold: (float) Variation threshold to use for coefficient to use in ``small_variations``.
    :return: (np.array) Resulting copula.
    """

    if target_dependence == 'comonotonicity':
        # Creating copula where each element is placed on the main diagonal
        target = np.array([[i / n_obs, i / n_obs] for i in range(n_obs)])

    elif target_dependence == 'countermonotonicity':
        # Creating copula where each element is placed on the counterdiagonal
        target = np.array([[i / n_obs, (n_obs - i) / n_obs] for i in range(n_obs)])

    elif target_dependence == 'gaussian':
        # Parameters to use when creating Gaussian copula
        mean = [0, 0]
        cov = [[1, gauss_corr],
               [gauss_corr, 1]]

        # Creating a set of observations to transform to copula
        target = np.random.multivariate_normal(mean, cov, n_obs)

        # Ranking observations - getting copula as a result
        target.T[0] = ss.rankdata(target.T[0]) / len(target.T[0])
        target.T[1] = ss.rankdata(target.T[1]) / len(target.T[1])

    elif target_dependence == 'positive_negative':
        # Creating copula where each even element is on the counterdiagonal and each odd is on the main diagonal
        target = np.array([[i / n_obs,
                            ((i % 2) * i + ((i + 1) % 2) * (n_obs - i)) / n_obs] for i in range(n_obs)])

    elif target_dependence == 'different_variations':
        # Creating copula where each even element is on the upper triangle odd is on the lower triangle
        target = np.array([[i / n_obs,
                            (abs(n_obs - ((i + 1) % 2) * i) - abs(n_obs - (i % 2) * i)) / n_obs] for i in range(n_obs)])

    elif target_dependence == 'small_variations':
        # Number of observations to be in the correlated part and outside of it
        obs_out = int(n_obs * (1 - var_threshold) / 2)
        obs_corr = n_obs - 2 * obs_out

        # The left part of copula consists of uncorrelated elements
        target_1 = np.array([[i / n_obs, ((i % 2) * np.random.uniform(0, obs_out) +
                                          ((i + 1) % 2) * np.random.uniform(n_obs - obs_out, n_obs)) / 5000]
                             for i in range(obs_out)])

        # Center part of copula consists of correlated elements

        # Parameters for the center part
        mean = [0, 0]
        cov = [[1, gauss_corr],
               [gauss_corr, 1]]

        # The center part observations created
        target_2 = np.random.multivariate_normal(mean, cov, obs_corr)

        # Ranked
        target_2.T[0] = ss.rankdata(target_2.T[0]) / len(target_2.T[0])
        target_2.T[1] = ss.rankdata(target_2.T[1]) / len(target_2.T[1])

        # And scaled
        target_2.T[1] = target_2.T[1] * var_threshold + (1 - var_threshold) / 2
        target_2.T[0] = target_2.T[0] * var_threshold + (1 - var_threshold) / 2

        # The right part of copula consists of uncorrelated elements
        target_3 = np.array([[i / n_obs, ((i % 2) * np.random.uniform(0, obs_out) +
                                          ((i + 1) % 2) * np.random.uniform(n_obs - obs_out, n_obs)) / 5000]
                             for i in range(n_obs - obs_out, n_obs)])

        # Combining parts of the copula
        target = np.concatenate((target_1, target_2, target_3), axis=0)

    else:
        raise Exception('This type of target dependence is not supported')

    return target
