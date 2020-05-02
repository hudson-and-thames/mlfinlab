# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class PassiveAggressiveMeanReversion(OLPS):
    """
    This class implements the Passive Aggressive Mean Reversion strategy. It is reproduced with
    modification from the following paper:
    'Li, B., Zhao, P., Hoi, S.C., & Gopalkrishnan, V. (2012). PAMR: Passive aggressive mean
    reversion strategy for portfolio selection. Machine Learning, 87, 221-258.
    <https://link.springer.com/content/pdf/10.1007%2Fs10994-012-5281-z.pdf>_'

    Passive Aggressive Mean Reversion strategy switches between a passive and an aggressive mean
    reversion strategy based on epsilon, a measure of sensitivty to the market, and hyperparameter C,
    which denotes the aggressiveness of reverting to a partciular strategy.
    """

    def __init__(self, epsilon, agg, optimization_method):
        """
        Initializes Passive Aggressive Mean Reversionw ith the given epsilon, aggressiveness,
        and optimzation method.

        :param epsilon: (float) Sensitivity to the market.
        :param agg: (float) Aggressiveness to mean reversion.
        :param optimization_method: (int) 0 for PAMR, 1 for PAMR-1, 2 for PAMR-2
        """
        self.epsilon = epsilon
        self.agg = agg
        self.optimization_method = optimization_method
        super().__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices.
        """
        super(PassiveAggressiveMeanReversion, self)._initialize(asset_prices, weights, resample_by)

        # Check that epsilon is greater than or equal to 0.
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("Epsilon values must be between 0 and 1")

        # Check that optimization method is either 1 or 2.
        if self.optimization_method not in [0, 1, 2]:
            raise ValueError("Optimization method must be either 0, 1, or 2.")

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return new_weights: (np.array) Predicted weights.
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
        # calculate the norm of the adjusted market change.
        diff_norm = np.linalg.norm(adjusted_market_change)

        # PAMR method looks to passively perform mean reversion.
        if self.optimization_method == 0:
            tau = loss / (diff_norm ** 2)
        # PAMR-1 introduces a slack variable for tradeoff between epsilon and C.
        elif self.optimization_method == 1:
            tau = min(self.agg, loss / (diff_norm ** 2))
        # PAMR-2 introduces a quadratic slack variable for tradeoff between epsilon and C.
        elif self.optimization_method == 2:
            tau = loss / (diff_norm ** 2 + 1 / (2 * self.agg))
        # Calculate new weights.
        new_weights = self.weights - tau * adjusted_market_change
        # Project new weights to simplex domain.
        new_weights = self._simplex_projection(new_weights)
        return new_weights
