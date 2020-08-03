# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.base import OLPS


class CORN(OLPS):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy. It is
    reproduced with modification from the following paper:
    `Li, B., Hoi, S.C., & Gopalkrishnan, V. (2011). CORN: Correlation-driven nonparametric
    learning approach for portfolio selection. ACM TIST, 2,
    21:1-21:29. <https://dl.acm.org/doi/abs/10.1145/1961189.1961193>`_

    Correlation Driven Nonparametric Learning finds similar windows from the past and looks to
    create portfolio weights that will maximize returns in the similar sets.
    """

    def __init__(self, window, rho):
        """
        Initializes Correlation Driven Nonparametric Learning with the given window and rho value.

        :param window: (int) Number of windows to look back for similarity sets. Windows can be set
                             to any values but typically work well in a shorter term of [1, 7].
        :param rho: (float) Threshold for similarity with range of [-1, 1].
                            Lower rho values will classify more periods as being similar, and higher
                            values will be more strict on identifying a period as similarly correlated.
                            Rho values between [0, 0.2] typically had higher results.
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

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """

        pass

    def _fast_optimize(self, optimize_array):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) Relative returns of the assets for a given time period.
        :return: (np.array) Weights that maximize the returns for the given array.
        """

        pass

    def _calc_rolling_corr_coef(self):
        """
        Calculates the rolling correlation coefficient for a given relative return and window

        :return: (np.array) Rolling correlation coefficient over a given window.
        """

        pass
