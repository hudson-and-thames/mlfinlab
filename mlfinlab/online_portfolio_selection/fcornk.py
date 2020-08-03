# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.up import UP
from mlfinlab.online_portfolio_selection.fcorn import FCORN


class FCORNK(UP):
    """
    This class implements the Functional Correlation Driven Nonparametric Learning - K strategy. It
    is reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Functional Correlation Driven Nonparametric Learning - K formulates a number of FCORN experts and
    tracks the experts performance over time. Each period, the strategy decides to allocate capital
    to the top-k experts until the previous time period. This strategy takes an ensemble approach to
    the top performing experts.
    """

    def __init__(self, window, rho, lambd, k):
        """
        Initializes Functional Correlation Driven Nonparametric Learning - K with the given number
        of window, rho, lambda, and k experts.

        :param window: (int) Number of windows to look back for similarity sets. Generates
                             experts with range of [1, 2, ..., w]. The window ranges typically work
                             well with shorter terms of [1, 7].
        :param rho: (int) Number of rho values for threshold. Generates experts with range of
                          [0, 1, (rho-1)/rho]. Higher rho values allow for a greater coverage of
                          possible parameters, but it will slow down the calculations. Rho ranges
                          typically work well with [3, 5].
        :param lambd: (int) Number of scale factors for sigmoid function. Generates experts with range
                            of [1, 10, 10 ** (lambd - 1)]. Higher lambd values allow for a greater
                            coverage of possible parameters, but it will slow down the calculations.
                            Lambd ranges typically work well with [1, 3].
        :param k: (int) Number of top-k experts. K values have range of [1, window * rho]. Higher
                        number of experts gives a higher exposure to a number of strategies. K value of
                        1 or 2 had the best results as some parameters significantly outperform others.
        """

        pass

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                  'M' for Month. The inputs are based on pandas' resample method.
        """

        pass

    def _generate_experts(self):
        """
        Generates window * rho experts from window of [1, w], rho of [0, (rho - 1) / rho], and
        lambd of [1, 10 ** (lambd-1)].
        """

        pass
