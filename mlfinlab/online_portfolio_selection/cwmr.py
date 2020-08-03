# pylint: disable=missing-module-docstring
import numpy as np
from scipy.stats import norm
from mlfinlab.online_portfolio_selection.base import OLPS


class CWMR(OLPS):
    """
    This class implements the Confidence Weighted Mean Reversion strategy. It is reproduced with
    modification from the following paper:
    `Li, B., Hoi, S.C., Zhao, P. & Gopalkrishnan, V.. (2011). Confidence Weighted Mean Reversion
    Strategy for On-Line Portfolio Selection. Proceedings of the Fourteenth International
    Conference on Artificial Intelligence and Statistics, in PMLR 15:434-442.
    <https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3292&context=sis_research>`_

    Confidence Weighted Mean Reversion exploits both the popular mean reversion techniques and
    second-order information to model weights as a gaussian distribution.
    """

    def __init__(self, confidence, epsilon, method='var'):
        """
        Initializes Confidence Weighted Mean Reversion with the given confidence, epsilon, and method.

        :param confidence: (float) Confidence parameter with range of [0, 1]. CWMR is extremely
                                   sensitive to parameters, and it is unpredictable how the confidence
                                   value directly affects the results. Typically the extreme values of
                                   0 and 1 have the highest returns. A low value indicates a wider
                                   range of selection for weights.
        :param epsilon: (float) Mean reversion parameter with range of [0, 1]. CWMR is extremely
                                sensitive to parameters, and it is unpredictable how the epsilon value
                                directly affects the results. Typically the extreme values of 0 and
                                1 have the highest returns.
        :param method: (string) Variance update method. Choose 'var' for variance and 'sd' for
                                standard deviation. Both methods are dependant on the confidence and
                                epsilon parameters more than the method itself.
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

    def _first_weight(self, weights):
        """
        Returns the first weight of the given portfolio. If the first weight is not given,
        initialize weights to uniform weights.

        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :return: (np.array) First portfolio weight.
        """

        pass

    def _calculate_lambd(self, new_m, new_v, new_w, mean_x):
        """
        Calculate lambd, the lagrangian multiplier, for the Confidence Weighted Mean Reversion
        update method.

        :param new_m: (float) Dot product of relative returns and current weights.
        :param new_v: (float) The variance of current relative returns.
        :param new_w: (float) The weighted variance of relative returns.
        :param mean_x: (np.array) Weighted average of the variance.
        :return: (float) Lagrangian multiplier for the problem.
        """

        pass
