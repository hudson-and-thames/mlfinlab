# pylint: disable=missing-module-docstring
import cvxpy as cp
from mlfinlab.online_portfolio_selection import OLPS


class FTL(OLPS):
    """
    This class implements Follow the Leader strategy.
    """

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        return self.optimize(_relative_return[:_time], _solver=cp.SCS)
