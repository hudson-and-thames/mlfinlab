# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.up import UP
from mlfinlab.online_portfolio_selection.corn import CORN


class CORNK(UP):
    """
    This class implements the Correlation Driven Nonparametric Learning - K strategy. It is
    reproduced with modification from the following paper:
    `Li, B., Hoi, S.C., & Gopalkrishnan, V. (2011). CORN: Correlation-driven nonparametric
    learning approach for portfolio selection. ACM TIST, 2,
    21:1-21:29. <https://dl.acm.org/doi/abs/10.1145/1961189.1961193>`_

    Correlation Driven Nonparametric Learning - K formulates a number of experts and tracks the
    experts performance over time. Each period, the strategy decides to allocate capital to
    the top-k experts until the previous time period. This strategy takes an ensemble approach to
    the top performing experts.
    """

    def __init__(self, window, rho, k):
        """
        Initializes Correlation Driven Nonparametric Learning - K with the given number of
        windows, rho values, and k experts.

        :param window: (int) Number of windows to look back for similarity sets. Generates
                             experts with range of [1, 2, ..., w]. The window ranges typically work
                             well with shorter terms of [1, 7].
        :param rho: (int) Number of rho values for threshold. Generates experts with range of
                          [0, 1, (rho-1)/rho]. Higher rho values allow for a greater coverage of
                          possible parameters, but it will slow down the calculations. Rho ranges
                          typically work well with [2, 7].
        :param k: (int) Number of top-k experts. K values have range of [1, window * rho]. Higher
                        number of experts gives a higher exposure to a number of strategies. K value of
                        1 or 2 had the best results as some parameters significantly outperform others.
        """

        pass

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                  'M' for Month. The inputs are based on pandas' resample method.
        """

        pass

    def _generate_experts(self):
        """
        Generates window * rho experts from window of [1, w] and rho of [0, (rho - 1) / rho].
        """

        pass
