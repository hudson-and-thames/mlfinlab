# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS


class RMR(OLPS):
    """
    This class implements the Confidence Weighted Mean Reversion strategy. It is reproduced with
    modification from the following paper:
    `D. Huang, J. Zhou, B. Li, S. C. H. Hoi and S. Zhou, "Robust Median Reversion Strategy for
    Online Portfolio Selection," in IEEE Transactions on Knowledge and Data Engineering, vol. 28,
    no. 9, pp. 2480-2493, 1 Sept. 2016. <https://www.ijcai.org/Proceedings/13/Papers/296.pdf>`_

    Robust Median Reversion uses a L1-median of historical prices to predict the next time's
    price relative returns. The new weights is then regularized to retain previous portfolio
    information but also seeks to maximize returns from the predicted window.
    """

    def __init__(self, epsilon, n_iteration, window, tau=0.001):
        """
        Initializes Robust Median Reversion with the given epsilon, n_iteration, window, and tau values.

        :param epsilon: (float) Reversion threshold with range [1, inf). Values of [15, 25] had the
                                highest returns for the original dataset provided by the authors.
        :param n_iteration: (int) Maximum number of iterations with range [2, inf). Iteration of 200
                                  produced a adequate balance between performance and computational
                                  time for the strategy.
        :param window: (int) Size of window with range [2, inf). Typically window of 2, 7, or 21
                             produced the highest results. This parameter depends on the underlying
                             data's price movements and mean reversion tendencies.
        :param tau: (float) Toleration level with range [0, 1). It is suggested to keep the toleration
                            level at 0.001 for computational efficiency.
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

    def _calculate_predicted_relatives(self, time):
        # pylint: disable=unsubscriptable-object
        """
        Calculates the predicted relatives using l1 median.

        :param time: (int) Current time.
        :return: (np.array) Predicted relatives using l1 median.
        """

        pass

    @staticmethod
    def _transform(old_mu, price_window):
        """
        Calculates L1 median approximation by using the Modified Weiszfeld Algorithm.

        :param old_mu: (np.array) Current value of the predicted median value.
        :param price_window: (np.array) A window of prices provided by the user.
        :return: (np.array) New updated l1 median approximation.
        """

        pass