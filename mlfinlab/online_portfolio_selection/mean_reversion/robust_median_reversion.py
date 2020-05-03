# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class RobustMedianReversion(OLPS):
    """
    This class implements the Confidence Weighted Mean Reversion strategy. It is reproduced with
    modification from the following paper:
    `D. Huang, J. Zhou, B. Li, S. C. H. Hoi and S. Zhou, "Robust Median Reversion Strategy for
    Online Portfolio Selection," in IEEE Transactions on Knowledge and Data Engineering, vol. 28,
    no. 9, pp. 2480-2493, 1 Sept. 2016.<https://www.ijcai.org/Proceedings/13/Papers/296.pdf>`_

    Robust Median Reversion uses a L1-median of historical prices to predict the next time's
    price relative returns. The new weights is then regularized to retain previous portfolio
    information but also seeks to maximize returns from the predicted window.
    """

    def __init__(self, epsilon, n_iteration, window, tau):
        """
        Initializes Robust Median Reversion with the given epsilon, window, and tau values.

        :param epsilon: (float) Reversion threshold. > 1
        :param n_iteration: (int) Maximum number of iterations.
        :param window: (int) Size of window.
        :param tau: (float) Toleration level.
        """
        self.epsilon = epsilon
        self.n_iteration = n_iteration
        self.window = window
        self.tau = tau
        self.l1_median = None  # (np.array) L1 Median of prices.
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices.
        """
        super(RobustMedianReversion, self)._initialize(asset_prices, weights, resample_by)

        # Check that epsilon value is correct.
        if self.epsilon < 1:
            raise ValueError("Epsilon values must be greater than 1.")

        # Check that the n_iteration is an integer.
        if type(self.n_iteration) != int:
            raise ValueError("Number of iterations must be an integer.")

        # Check that the window value is greater or equal to 1.
        if self.n_iteration >= 1:
            raise ValueError("Window must be greater or equal to 1.")

        # Check that the window value is an integer.
        if type(self.window) != int:
            raise ValueError("Window must be an integer.")

        # Check that the window value is greater or equal to 2.
        if self.window < 2:
            raise ValueError("Window must be greater or equal to 2.")

        self.l1_median = self._calculate_l1_median()


    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return new_weights: (np.array) Predicted weights.
        """

    def _calculate_l1_median(self):
        rolling_median = self.asset_prices.rolling(self.window).median()

    def _iterate_rolling(self, median_price, curr_price):
        old_median = median_price
        for iters in range(1, self.n_iteration):
            new_median = _helper(old_median)
            if np.linalg.norm(old_median - new_median, ord=1) <= self.tau * np.linalg.norm(new_median):
                break
        return new_median / curr_price

    # @staticmethod
    # def _helper(old_median):
    #
    #
    # y = np.mean(X, 0)
    #
    # while True:
    #     D = cdist(X, [y])
    #     nonzeros = (D != 0)[:, 0]
    #
    #     Dinv = 1 / D[nonzeros]
    #     Dinvs = np.sum(Dinv)
    #     W = Dinv / Dinvs
    #     T = np.sum(W * X[nonzeros], 0)
    #     num_zeros = len(X) - np.sum(nonzeros)
    #     if num_zeros == 0:
    #         y1 = T
    #     elif num_zeros == len(X):
    #         return y
    #     else:
    #         R = (T - y) * Dinvs
    #         r = np.linalg.norm(R)
    #         rinv = 0 if r == 0 else num_zeros / r
    #         y1 = max(0, 1 - rinv) * T + min(1, rinv) * y
