# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class BESTSTOCK(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    The Best Stock strategy represents the best performing asset over the period in hindsight.
    """

    def first_weight(self, _weights):
        """
        Sets the initial weight to the best performing stock over the entire time period

        :param _weights: (np.array) Given weights do not matter as new weights will be calculated
        :return new_weight: (np.array) weight that allocates one to the best performing asset
        """
        # index of stock that increased the most
        best_idx = np.argmax(self.relative_return.cumprod(axis=0)[-1])
        # initialize array of zeros
        new_weight = np.zeros(self.number_of_assets)
        # assign one to best performing stock
        new_weight[best_idx] = 1
        return new_weight
