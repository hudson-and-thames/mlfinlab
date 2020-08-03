# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS


class BestStock(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the
    following paper:
    `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

    The Best Stock strategy represents the best performing asset over the period in hindsight.
    """

    def _first_weight(self, weights):
        """
        Sets the initial weight to the best performing stock over the entire time period.

        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :return: (np.array) Weights that allocate to the best performing asset.
        """

        pass
