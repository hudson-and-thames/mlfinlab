# pylint: disable=missing-module-docstring
import numpy as np
import scipy.optimize as opt
from mlfinlab.online_portfolio_selection.ftl import FTL


class FTRL(FTL):
    """
    This class implements the Follow the Regularized Leader strategy. It is reproduced with
    modification from the following paper:
    `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

    Follow the Regularized Leader strategy directly tracks the Best Constant Rebalanced Portfolio
    until the previous period with an additional regularization term
    """

    def __init__(self, beta):
        """
        Initializes Follow the Regularized Leader with a beta constant term.

        :param beta: (float) Constant to the regularization term. Typical ranges for interesting
                             results include [0, 0.2], 1, and any high values. Low beta
                             FTRL strategies are identical to FTL, and high beta indicates more
                             regularization to return a uniform CRP.
        """
        super(FTRL, self).__init__()
        self.beta = beta

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
        # Add additional l2 regularization term for the weights for calculation.
        def _objective(weight):
            return -np.sum(np.log(np.dot(optimize_array, weight))) + self.beta * np.linalg.norm(weight) / 2

        # Weight bounds.
        bounds = tuple((0.0, 1.0) for asset in range(self.number_of_assets))

        # Sum of weights is 1.
        const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        problem = opt.minimize(_objective, weights, method='SLSQP', bounds=bounds, constraints=const)
        return problem.x
