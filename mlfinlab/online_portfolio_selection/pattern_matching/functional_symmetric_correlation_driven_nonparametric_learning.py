# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning import CorrelationDrivenNonparametricLearning


class FCORN(CorrelationDrivenNonparametricLearning):
    """
    This class implements the Symmetric Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Functional Correlation Driven Nonparametric Learning is an extension of the SCORN strategy.
    Unlike SCORN, that sets a certain threshold for the objective function, FCORN places an
    arbitrary weight to all relative returns period to emphasize certain weights. If the given
    lambda value approaches infinity, then FCORN returns the same values as the SCORN strategy.
    """

    def __init__(self, window, rho, lambd):
        """
        Initializes Functional Correlation Driven Nonparametric Learning with the given window,
        rho, and lambd value.

        :param window: (int) Number of windows to look back for similarity sets.
        :param rho: (float) Threshold for similarity.
        :param lambd:
        """
        self.lambd = lambd
        super().__init__(window=window, rho=rho)

    def _update_weight(self, _weights, _relative_return, _time):
        """
        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        similar_set = []
        opposite_set = []
        new_weights = self._uniform_weight(self.number_of_assets)
        if _time - 1 > self.window:
            activation_fn = np.zeros(self.final_number_of_time)
            for i in range(self.window + 1, _time - 1):
                c = self.corr_coef[i - 1][_time - 1]
                if c >= 0:
                    activation_fn[i] = self.sigmoid(-self.lamb * (c - self.rho))
                else:
                    activation_fn[i] = self.sigmoid(-self.lamb * (c + self.rho))
            new_weights = self._optimize(_relative_return, activation_fn)
        return new_weights

    def sigmoid(val):
        """
        Generates the resulting sigmoid function

        :param val: (float) input for the sigmoid function
        :return sig: (float) sigmoid(x)
        """
        res = 1 / (1 + np.exp(-val))
        return res
