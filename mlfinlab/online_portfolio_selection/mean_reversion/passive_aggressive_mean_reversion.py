# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class PassiveAggressiveMeanReversion(OLPS):
    """
    This class implements the Passive Aggressive Mean Reversion strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    Passive Aggressive Mean Reversion strategy switches between a passive and an aggressive mean reversion strategy based on epsilon,
    a measure of sensitivty to the market. Hyperparameter C denotes the aggressiveness of reverting to a partciular strategy.
    """

    def __init__(self,
                 epsilon,
                 agg,
                 optimization_method):
        """

        :param epsilon: (float) measure of sensitivity to the market
        :param agg: (float) measure of aggressiveness to mean reversion
        :param optimization_method: (int) 0 for PAMR, 1 for PAMR-1, 2 for PAMR-2
        """
        # check that sensitivity is within [0,1]
        self.epsilon = epsilon
        self.agg = agg
        self.optimization_method = optimization_method
        super().__init__()

    def update_weight(self,
                      _time):
        """
        Updates portfolio weights

        :param _time: (int) current time period
        :return (None) sets new weights to be the same as old weights
        """
        # first prediction should be self.weights
        if _time == 0:
            return self.weights
        # calculation prep
        _past_relative_return = self.relative_return[_time]
        loss = max(0, np.dot(self.weights, _past_relative_return) - self.epsilon)
        adjusted_market_change = _past_relative_return - self.uniform_weight() * np.mean(_past_relative_return)
        diff_norm = np.linalg.norm(adjusted_market_change)

        # PAMR
        # looks to passively perform mean reversion
        if self.optimization_method == 0:
            tau = loss / (diff_norm ** 2)
        # PAMR-1
        # introduces a slack variable for tradeoff between epsilon and C
        elif self.optimization_method == 1:
            tau = min(self.agg, loss / (diff_norm ** 2))
        # PAMR-2
        # introduces a quadratic slack variable for tradeoff between epsilon and C
        elif self.optimization_method == 2:
            tau = loss / (diff_norm ** 2 + 1 / (2 * self.agg))

        new_weights = self.weights - tau * adjusted_market_change
        # if not in simplex domain
        return self.simplex_projection(new_weights)
