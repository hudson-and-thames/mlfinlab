# pylint: disable=missing-module-docstring
import cvxpy as cp
from mlfinlab.online_portfolio_selection import OLPS


class FollowTheLeader(OLPS):
    """
    This class implements the Follow the Leader strategy. It is reproduced with modification from
    the following paper: Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM
    Comput. Surv. V, N, Article A (December YEAR), 33 pages. <https://arxiv.org/abs/1212.2129>.

    Follow the Leader strategy directly tracks the Best Constant Rebalanced Portfolio until the
    previous period.
    """

    def update_weight(self, time):
        """
        Predicts the next time's portfolio weight by following the Best Constant Rebalanced
        Portfolio.

        :param time: (int) current time period.
        :return new_weights: (np.array) new portfolio weights by calculating the BCRP until the
                                        last time period.
        """
        # for time 0, return initial weights
        if time == 0:
            return self.weights
        # calculates bcrp weights until the last window
        new_weights = self.optimize(self.relative_return[:time+1], solver=cp.SCS)
        return new_weights
