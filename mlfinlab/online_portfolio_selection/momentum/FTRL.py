# pylint: disable=missing-module-docstring
import numpy as np
import cvxpy as cp
from mlfinlab.online_portfolio_selection import FTL


class FTRL(FTL):
    """
    This class implements the Follow the Regularized Leader strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    Follow the Regularized Leader strategy directly tracks the BCRP until the previous period with an additional regularization term
    """
    def __init__(self, beta=0.1):
        """
        Set beta, a regularization term

        :param beta: (float) constant multiplied to regularization term
        """
        self.beta = beta
        super(FTRL, self).__init__()

    def update_weight(self, _weights, _relative_return, _time):
        """
        Updates weight to find the BCRP and regularization adjusted weights until the last time period

        :param _weights: (np.array) portfolio weights of the previous period
        :param _relative_return: (np.array) relative returns of all period
        :param _time: (int) current time period
        :return:
        """
        return self.optimize(_relative_return[:_time], _solver=cp.SCS)

    # optimize the weight that maximizes the returns
    def optimize(self, _optimize_array, _solver=None):
        length_of_time = _optimize_array.shape[0]
        number_of_assets = _optimize_array.shape[1]
        if length_of_time == 1:
            best_idx = np.argmax(_optimize_array)
            weight = np.zeros(number_of_assets)
            weight[best_idx] = 1
            return weight

        weights = cp.Variable(self.number_of_assets)
        # added additiona l2 regularization term for the weights for calculation
        portfolio_return = cp.sum(cp.log(_optimize_array * weights)) - self.beta * cp.norm(weights) / 2

        # Optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [
                cp.sum(weights) == 1,
                cp.min(weights) >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
                objective=allocation_objective,
                constraints=allocation_constraints
        )
        if _solver:
            problem.solve(solver=_solver)
        else:
            problem.solve()
        return weights.value
