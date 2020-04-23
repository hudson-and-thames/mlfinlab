# pylint: disable=missing-module-docstring
import cvxpy as cp
from mlfinlab.online_portfolio_selection import OLPS


class FollowTheLeader(OLPS):
    """
    This class implements the Follow the Leader strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    Follow the Leader strategy directly tracks the Best Constant Rebalanced Portfolio until the previous period.
    """

    def update_weight(self,
                      _time):
        """
        Updates weight to find the Best Constant Rebalanced Portfolio weights until the last time period

        :param _time: (int) current time period
        :return:
        """
        # first return initial weights
        if _time == 0:
            return self.weights
        # calculates bcrp weights until the last window
        new_weights = self.optimize(self.relative_return[:_time], _solver=cp.SCS)
        return new_weights
