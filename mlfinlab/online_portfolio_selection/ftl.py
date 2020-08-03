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

        pass

    def _fast_optimize(self, optimize_array):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) Relative returns of the assets for a given time period.
        :return: (np.array) Weights that maximize the returns for the given array.
        """

        pass
