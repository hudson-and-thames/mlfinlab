# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.corn import CORN


class SCORN(CORN):
    """
    This class implements the Symmetric Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Symmetric Correlation Driven Nonparametric Learning is an extension of the CORN strategy
    proposed by Yang Wang and Dong Wang. SCORN looks to not only maximize the returns for the similar
    periods but also minimize the losses from the negatively correlated periods.
    """

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """

        pass

    def _scorn_optimize(self, similar, opposite):
        """
        Calculates weights that maximize returns over the given array.

        :param similar: (np.array) Relative returns of similar periods.
        :param opposite: (np.array) Relative returns of inversely similar periods.
        :return: (np.array) Weights that maximize the returns for the given array.
        """

        pass
