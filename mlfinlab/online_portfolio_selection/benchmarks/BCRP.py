# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection import OLPS
import cvxpy as cp


class BCRP(OLPS):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def first_weight(self, _weights):
        """
        :param _weights: optimized first weight
        :return:
        """
        return self.optimize(self.final_relative_return, _solver=cp.SCS)
