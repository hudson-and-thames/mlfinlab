"""
Implementations of Optimal Transport dependence measure porposed by Dr. Marti et al. : https://arxiv.org/abs/1610.09659
And implemented in the blog post by Dr. Marti: https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html
"""
import numpy as np
import scipy.stats as ss
import ot


# pylint: disable=invalid-name

def get_optimal_transport_distance(x: np.array, y: np.array, normalize: bool = False) -> float:
    """
    Returns optimal transport distance between two vectors.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)
    :return: (float) Optimal transport distance.
    """

    return optimal_transport
