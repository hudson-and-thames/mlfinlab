"""
Implementations of Optimal Copula Transport dependence measure proposed by Marti et al. : https://arxiv.org/abs/1610.09659
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

    pass


def optimal_transport_dependence(x: np.array, y: np.array, target_dependence: str = 'comonotonicity',
                                 gaussian_corr: float = 0.7, var_threshold: float = 0.2) -> float:
    """
    Calculates optimal copula transport dependence between the empirical copula of the two vectors and a target copula.

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
    - ``v-shape`` - a copula that is seen with vol index vs. returns index: when returns of the index
      are extreme, vol is usually high, when returns small in absolute value, vol usually low.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param target_dependence: (str) Type of target dependence to use when measuring distance.
                                    (``comonotonicity`` by default)
    :param gaussian_corr: (float) Correlation coefficient to use when creating ``gaussian`` and
                                  ``small_variations`` copulas. [from 0 to 1] (0.7 by default)
    :param var_threshold: (float) Variation threshold to use for coefficient to use in ``small_variations``.
                                  Sets the relative area of correlation in a copula. [from 0 to 1] (0.2 by default)
    :return: (float) Optimal copula transport dependence.
    """

    pass


def _compute_copula_ot_dependence(empirical: np.array, target: np.array, forget: np.array,
                                  n_obs: int) -> float:
    """
    Calculates optimal copula transport dependence measure.

    :param empirical: (np.array) Empirical copula.
    :param target: (np.array) Target copula.
    :param forget: (np.array) Forget copula.
    :param nb_obs: (int) Number of observations.
    :return: (float) Optimal copula transport dependence.
    """

    pass


def _create_target_copula(target_dependence: str, n_obs: int, gauss_corr: float,
                          var_threshold: float) -> np.array:
    """
    Creates target copula with given dependence and number of observations.

    :param target_dependence: (str) Type of dependence to use for copula creation.[``comonotonicity``,
                                    ``countermonotonicity``, ``gaussian``, ``positive_negative``,
                                    ``different_variations``, ``small_variations``, ``v-shape``]
    :param n_obs: (int) Number of observations to use for copula creation.
    :param gauss_corr: (float) Correlation coefficient to use when creating ``gaussian`` and
                                  ``small_variations`` copulas.
    :param var_threshold: (float) Variation threshold to use for coefficient to use in ``small_variations``.
    :return: (np.array) Resulting copula.
    """

    pass
