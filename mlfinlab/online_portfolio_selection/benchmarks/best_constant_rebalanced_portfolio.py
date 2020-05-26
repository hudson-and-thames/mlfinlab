# pylint: disable=missing-module-docstring
import cvxpy as cp
from mlfinlab.online_portfolio_selection.base import OLPS


class BCRP(OLPS):
    """
    This class implements the Best Constant Rebalanced Portfolio strategy. It is reproduced with
    modification from the following paper:
    `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

    Best Constant Rebalanced Portfolio rebalances to a set of weight that maximizes returns over a
    given time period. This strategy is implemented in hindsight and is not predictive.
    """

    def _first_weight(self, weights):
        """
        Returns the first weight of the given portfolio to be the Best Constant Rebalanced Portfolio
        in hindsight.

        :param weights: (np.array) Given weights by the user.
        :return: (np.array) Weights that maximize the returns.
        """
        # Use cp.SCS solver to speed up calculations.
        new_weights = self._optimize(self.relative_return, solver=cp.SCS)
        return new_weights
