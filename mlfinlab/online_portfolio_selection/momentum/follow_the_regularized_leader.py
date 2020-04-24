# pylint: disable=missing-module-docstring
import cvxpy as cp
from mlfinlab.online_portfolio_selection import FollowTheLeader


class FollowTheRegularizedLeader(FollowTheLeader):
    """
    This class implements the Follow the Regularized Leader strategy. It is reproduced with
    modification from the following paper: Li, B., Hoi, S. C.H., 2012. OnLine Portfolio
    Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR), 33 pages.
    <https://arxiv.org/abs/1212.2129>.

    Follow the Regularized Leader strategy directly tracks the Best Constant Rebalanced Portfolio until the previous
    period with an additional regularization term
    """

    def __init__(self, beta):
        """
        Initializes Follow the Regularized Leader with a beta constant term.

        :param beta: (float) a constant multiple to the regularization term.
        """
        super(FollowTheRegularizedLeader, self).__init__()
        self.beta = beta

    def optimize(self, optimize_array, solver=cp.SCS):
        """
        Calculates weights that maximize returns over a given optimize_array.

        :param optimize_array: (np.array) relative returns of the assets for a given time period.
        :param solver: (cp.SOLVER) set the solver to be a particular cvxpy solver.
        :return weights.value: (np.array) weights that maximize the returns for the given array.
        """
        weights = cp.Variable(self.number_of_assets)
        # added additional l2 regularization term for the weights for calculation
        returns = cp.sum(cp.log(optimize_array * weights)) - self.beta * cp.norm(weights) / 2

        # optimization objective and constraints
        allocation_objective = cp.Maximize(returns)
        allocation_constraints = [cp.sum(weights) == 1, cp.min(weights) >= 0]
        # define and solve the problem
        problem = cp.Problem(objective=allocation_objective, constraints=allocation_constraints)
        problem.solve(warm_start=True, solver=solver)
        return weights.value
