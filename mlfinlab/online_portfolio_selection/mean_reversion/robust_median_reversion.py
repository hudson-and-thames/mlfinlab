# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class RobustMedianReversion(OLPS):
    """
    This class implements the Confidence Weighted Mean Reversion strategy. It is reproduced with
    modification from the following paper:
    'D. Huang, J. Zhou, B. Li, S. C. H. Hoi and S. Zhou, "Robust Median Reversion Strategy for
    Online Portfolio Selection," in IEEE Transactions on Knowledge and Data Engineering, vol. 28,
    no. 9, pp. 2480-2493, 1 Sept. 2016.<https://www.ijcai.org/Proceedings/13/Papers/296.pdf>_'

    Robust Median Reversion uses a L1-median of historical prices to predict the next time's
    price relative returns. The new weights is then regularized to retain previous portfolio
    information but also seeks to maximize returns from the predicted window.
    """

    def __init__(self, epsilon, window, tau):
        """
        Initializes Robust Median Reversion with the given epsilon, window, and tau values.

        :param epsilon: (float) Reversion threshold. Must be greater or equal to 1.
        :param window: (int) Number of windows.
        :param tau: (float) Toleration level.
        """
        self.epsilon = epsilon
        self.window = window
        self.tau = tau
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
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("Epsilon values must be between 0 and 1.")

        # Check that the window value is greater or equal to 2.
        if type(self.window) != int:
            raise ValueError("Window must be an integer.")

        # Check that the window value is greater or equal to 2.
        if self.window < 2:
            raise ValueError("Window must be greater or equal to 2.")

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return new_weights: (np.array) Predicted weights.
        """
