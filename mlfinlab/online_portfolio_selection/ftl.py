# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.base import OLPS


class FTL(OLPS):
    """
    This class implements the Follow the Leader strategy. It is reproduced with modification from
    the following paper:
    `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

    Follow the Leader strategy directly tracks the Best Constant Rebalanced Portfolio until the
    previous period.
    """

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight by following the Best Constant Rebalanced
        Portfolio.

        :param time: (int) Current time period.
        :return: (np.array) New portfolio weights by calculating the BCRP until the
                                        last time period.
        """
        # For time 0, return initial weights.
        if time == 0:
            return self.weights
        # Calculate BCRP weights until the current window.
        new_weights = self._fast_optimize(self.relative_return[:time + 1])
        return new_weights

    def _fast_optimize(self, optimize_array):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) Relative returns of the assets for a given time period.
        :return: (np.array) Weights that maximize the returns for the given array.
        """
        # Initialize guess.
        weights = self._uniform_weight()

        # Use np.log and np.sum to make the cost function a convex function.
        # Multiplying continuous returns equates to summing over the log returns.
        def _objective(weight):
            return -np.sum(np.log(np.dot(optimize_array, weight)))

        # Weight bounds.
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))

        # Sum of weights is 1.
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        problem = opt.minimize(_objective, weights, method='SLSQP', bounds=bounds, constraints=const)
        return problem.x
