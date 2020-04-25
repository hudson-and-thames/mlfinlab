# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
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
        new_weights = self.fast_optimize(self.relative_return[:time+1])
        return new_weights

    def fast_optimize(self, optimize_array):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) relative returns of the assets for a given time period.
        :return: problem.x: (np.array) weights that maximize the returns for the given array.
        """
        # initial guess
        weights = self.uniform_weight()

        # use np.log and np.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        def objective(weight):
            return -np.sum(np.log(np.dot(optimize_array, weight)))

        # weight bounds
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))

        # sum of weights = 1
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        problem = opt.minimize(objective, weights, method='SLSQP', bounds=bounds, constraints=const)
        return problem.x
