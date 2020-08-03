# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.corn import CORN


class FCORN(CORN):
    """
    This class implements the Functional Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Functional Correlation Driven Nonparametric Learning is an extension of the SCORN strategy.
    Unlike SCORN that sets a certain threshold for the objective function, FCORN places an
    arbitrary weight to all relative returns period to emphasize certain weights. If the given
    lambda value approaches infinity, then FCORN returns the same values as the SCORN strategy.
    """

    def __init__(self, window, rho, lambd):
        """
        Initializes Functional Correlation Driven Nonparametric Learning with the given window,
        rho, and lambd value.

        :param window: (int) Number of windows to look back for similarity sets. Windows can be set
                             to any values but typically work well in a shorter term of [1, 7].
        :param rho: (float) Threshold for similarity. Rho should set in the range of [-1, 1].
                            Lower rho values will classify more periods as being similar, and higher
                            values will be more strict on identifying a period as similarly correlated.
                            Rho values between [0.4, 0.8] typically had higher results.
        :param lambd: (float) Scale factor for the sigmoid activation function. Lambd can be any
                              value. If lambd approaches infinity, FCORN will have the same results
                              as SCORN, and if lambd approaches negative infinity, FCORN produces
                              similar results as a Constant Rebalanced Portfolio. From experimental
                              results, lambd of 1 was optimal for some datasets.
        """

        pass

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """

        pass

    def _fcorn_optimize(self, activation_fn, relative_return):
        """
        Calculates weights that maximize returns over the given array.

        :param activation_fn: (np.array) Scaling factor for optimization equation.
        :param relative_return: (np.array) Array of relative returns until the current period
        :return: (np.array) Weights that maximize the returns for the given array.
        """

        pass

    @staticmethod
    def _sigmoid(val):
        """
        Generates the resulting sigmoid function

        :param val: (float) Input for
        :return: (float) sigmoid(x)
        """

        pass
