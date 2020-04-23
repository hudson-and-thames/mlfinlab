# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class PAMR(OLPS):
    """
    This class implements the Passive Aggressive Mean Reversion strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    Passive Aggressive Mean Reversion strategy
    """

    def __init__(self, epsilon=0.5, c=1, optimization_method=0):
        """

        :param epsilon: (float) measure of sensitivity to the market
        :param aggressiveness: (float) measure of aggressiveness to the market
        :param optimization_method:
        """
        # check that sensitivity is within [0,1]
        self.epsilon = epsilon
        self.c = c
        self.optimization_method = optimization_method
        super().__init__()

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        # calculation prep
        _past_relative_return = _relative_return[_time - 1]
        loss = max(0, np.dot(_weights, _past_relative_return) - self.epsilon)
        adjusted_market_change = _past_relative_return - self.uniform_weight(self.number_of_assets) * np.mean(
                _past_relative_return)
        diff_norm = np.linalg.norm(adjusted_market_change)

        # different optimization methods
        if self.optimization_method == 0:
            tau = loss / (diff_norm ** 2)
        elif self.optimization_method == 1:
            tau = min(self.c, loss / (diff_norm ** 2))
        elif self.optimization_method == 2:
            tau = loss / (diff_norm ** 2 + 1 / (2 * self.c))

        new_weights = _weights - tau * adjusted_market_change
        # if not in simplex domain
        return self.simplex_projection(new_weights)
