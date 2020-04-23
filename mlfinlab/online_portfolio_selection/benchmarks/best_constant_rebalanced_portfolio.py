# pylint: disable=missing-module-docstring
import cvxpy as cp
from mlfinlab.online_portfolio_selection import OLPS


class BestConstantRebalancedPortfolio(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    Best Constant Rebalanced Portfolio rebalances to a set weight that maximizes returns over a given time period.
    This strategy is implemented in hindsight and is not predictive.
    """

    def first_weight(self, _weights):
        """
        :param _weights: (np.array) given weights that do not affect the new weights
        :return new_weights: (np.array) weights that maximize the returns
        """
        # used cp.SCS solver to speed up calculations
        new_weights = self.optimize(self.relative_return, _solver=cp.SCS)
        return new_weights
