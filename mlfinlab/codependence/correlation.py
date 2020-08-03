"""
Correlation based distances and various modifications (angular, absolute, squared) described in Cornell lecture notes:
Codependence: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist


# pylint: disable=invalid-name


def angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns angular distance between two vectors. Angular distance is a slight modification of Pearson correlation which
    satisfies metric conditions.

    Formula used for calculation:

    Ang_Distance = (1/2 * (1 - Corr))^(1/2)

    Read Cornell lecture notes for more information about angular distance:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Angular distance.
    """

    pass


def absolute_angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns absolute angular distance between two vectors. It is a modification of angular distance where the absolute
    value of the Pearson correlation coefficient is used.

    Formula used for calculation:

    Abs_Ang_Distance = (1/2 * (1 - abs(Corr)))^(1/2)

    Read Cornell lecture notes for more information about absolute angular distance:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Absolute angular distance.
    """

    pass


def squared_angular_distance(x: np.array, y: np.array) -> float:
    """
    Returns squared angular distance between two vectors. It is a modification of angular distance where the square of
    Pearson correlation coefficient is used.

    Formula used for calculation:

    Squared_Ang_Distance = (1/2 * (1 - (Corr)^2))^(1/2)

    Read Cornell lecture notes for more information about squared angular distance:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Squared angular distance.
    """

    pass


def distance_correlation(x: np.array, y: np.array) -> float:
    """
    Returns distance correlation between two vectors. Distance correlation captures both linear and non-linear
    dependencies.

    Formula used for calculation:

    Distance_Corr[X, Y] = dCov[X, Y] / (dCov[X, X] * dCov[Y, Y])^(1/2)

    dCov[X, Y] is the average Hadamard product of the doubly-centered Euclidean distance matrices of X, Y.

    Read Cornell lecture notes for more information about distance correlation:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Distance correlation coefficient.
    """

    pass
