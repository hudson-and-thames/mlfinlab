# pylint: disable=missing-module-docstring
import cvxpy as cp
from mlfinlab.online_portfolio_selection import OLPS


class FTL(OLPS):
    """
    This class implements the Follow the Leader strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    Follow the Leader strategy directly tracks the BCRP until the previous period.
    """

    def update_weight(self, _weights, _relative_return, _time):
        """
        Updates weight to find the BCRP weights until the last time period

        :param _weights: (np.array) portfolio weights of the previous period
        :param _relative_return: (np.array) relative returns of all period
        :param _time: (int) current time period
        :return:
        """
        # calculates bcrp weights until the last window
        new_weights = self.optimize(_relative_return[:_time], _solver=cp.SCS)
        return new_weights
