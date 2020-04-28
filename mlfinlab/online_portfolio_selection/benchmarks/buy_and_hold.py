# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class BuyAndHold(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the
    following paper:
    'Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December YEAR), 33 pages. <https://arxiv.org/abs/1212.2129>'_

    The Buy and Hold strategy invests capital with an initial portfolio of weights and holds the
    portfolio until the end. The manager only buys the assets at the beginning of the first period
    and does not rebalance in subsequent periods.
    """

    def _update_weight(self, time):
        """
        Changes weights to adjust for underlying asset price changes.

        :param time: (int) Current time period.
        """
        # Adjust weights for price differences.
        # Weights will be normalized later in round_weights.
        return self.weights * self.relative_return[time]
