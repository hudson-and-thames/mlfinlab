# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS


class OLMAR(OLPS):
    """
    This class implements the Online Moving Average Reversion strategy. It is reproduced with
    modification from the following paper:
    `Li, Bin & Hoi, Steven. (2012). On-Line Portfolio
    Selection with Moving Average Reversion. Proceedings of the 29th International Conference on
    Machine Learning, ICML 2012. 1. <https://arxiv.org/pdf/1206.4626.pdf>`_

    Online Moving Average Reversion reverts to the SMA or EMA of the underlying assets based on
    the given threshold.
    """

    def __init__(self, reversion_method, epsilon, window=None, alpha=None):
        """
        Initializes Online Moving Average Reversion with the given reversion method, epsilon,
        window, and alpha.

        :param reversion_method: (int) 1 for SMA, 2 for EWA. Typically 2 has higher returns than 1,
                                       but this is dependant on the dataset.
        :param epsilon: (float) Reversion threshold with range [1, inf). OLMAR methods typically do
                                not rely on epsilon values as it is more dependant on the window and
                                alpha value; however, epsilon of 20 seems to be a safe choice to
                                avoid extreme values.
        :param window: (int) Number of windows to calculate Simple Moving Average with range [1, inf).
                             This parameter depends on the data. If the data has a shorter mean reversion
                             trends, then window of 6 can have high returns. If the data has a longer
                             mean reversion trend, then values close to 21 can have high returns.
        :param alpha: (float) Exponential ratio for Exponentially Weighted Average with range (0, 1).
                              Larger alpha value indicates more importance placed on recent prices,
                              and smaller alpha value indicates more importance placed on the past.
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

    def _calculate_rolling_moving_average(self):
        """
        Calculates the rolling moving average for Online Moving Average Reversion.

        :return: (np.array) Rolling moving average for the given reversion method.
        """

        pass
