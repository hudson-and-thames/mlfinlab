# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS


class PAMR(OLPS):
    """
    This class implements the Passive Aggressive Mean Reversion strategy. It is reproduced with
    modification from the following paper:
    `Li, B., Zhao, P., Hoi, S.C., & Gopalkrishnan, V. (2012). PAMR: Passive aggressive mean
    reversion strategy for portfolio selection. Machine Learning, 87, 221-258.
    <https://link.springer.com/content/pdf/10.1007%2Fs10994-012-5281-z.pdf>`_

    Passive Aggressive Mean Reversion strategy switches between a passive and an aggressive mean
    reversion strategy based on epsilon, a measure of sensitivity to the market,
    and hyperparameter C, which denotes the aggressiveness of reverting to a particular strategy.
    """

    def __init__(self, optimization_method, epsilon=0.5, agg=10):
        """
        Initializes Passive Aggressive Mean Reversion with the given epsilon, aggressiveness,
        and optimzation method.

        :param optimization_method: (int) 0 for PAMR, 1 for PAMR-1, 2 for PAMR-2. All three methods
                                          tend to return similar values.
        :param epsilon: (float) Sensitivity to the market with range of [0, inf). Because the epsilon
                                is considered a threshold and daily returns typically are around 1.
                                It is suggested to keep the range of [0, 1.5]. Typically, the returns
                                are highest with either a value of 0 or 1 for epsilon. 0 indicates
                                active mean reversion for all periods, and 1 indicates passive mean
                                reversion for daily returns below 1.
        :param agg: (float) Aggressiveness to mean reversion with range [0, inf). Aggressiveness
                            does not have much of an impact as epsilon. Typically, 100 has the highest
                            returns for PAMR-1 and 10000 has the highest returns for PAMR-2.
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
