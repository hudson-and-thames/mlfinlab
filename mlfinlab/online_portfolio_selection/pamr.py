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
        self.epsilon = epsilon
        self.agg = agg
        self.optimization_method = optimization_method
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        """
        super(PAMR, self)._initialize(asset_prices, weights, resample_by)

        # Check that epsilon is greater than 0.
        if self.epsilon < 0:
            raise ValueError("Epsilon values must be greater than 0")

        # Check that aggressiveness is greater than 0.
        if self.agg < 0:
            raise ValueError("Aggressiveness values must be greater than 0")

        # Check that optimization method is either 0, 1, or 2.
        if self.optimization_method not in [0, 1, 2]:
            raise ValueError("Optimization method must be either 0, 1, or 2.")

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Predicted weights.
        """
        # First prediction returns the same weights.
        if time == 0:
            return self.weights

        # Prepare for calculations.
        current_relative_return = self.relative_return[time]

        # Calculate loss function.
        loss = max(0, np.dot(self.weights, current_relative_return) - self.epsilon)

        # Calculate the adjusted market change.
        adjusted_market_change = current_relative_return - self._uniform_weight() \
                                 * np.mean(current_relative_return)

        # Calculate the norm of the adjusted market change.
        diff_norm = np.linalg.norm(adjusted_market_change)

        # PAMR method looks to passively perform mean reversion.
        if self.optimization_method == 0:
            tau = loss / (diff_norm ** 2)

        # PAMR-1 introduces a slack variable for tradeoff between epsilon and C.
        elif self.optimization_method == 1:
            tau = min(self.agg, loss / (diff_norm ** 2))

        # PAMR-2 introduces a quadratic slack variable for tradeoff between epsilon and C.
        else:
            tau = loss / (diff_norm ** 2 + 1 / (2 * self.agg))

        # Calculate new weights.
        new_weights = self.weights - tau * adjusted_market_change

        # Project new weights to simplex domain.
        new_weights = self._simplex_projection(new_weights)
        return new_weights
