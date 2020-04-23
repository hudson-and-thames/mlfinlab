# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class ConstantRebalancedPortfolio(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/2512962.

    Constant Rebalanced Portfolio rebalances to a given weight for each time period.

    :ivar weights: (np.array) any initial weights that the user wants to use
    """
    def __init__(self, weights=None):
        super(ConstantRebalancedPortfolio, self).__init__()
        self.weights = weights

    def first_weight(self, _weights):
        """
        Returns the first weight of the given portfolio

        :param _weights: (list/np.array/pd.Dataframe) weights given by the user, none if not initialized
        :return _weights: (np.array) returns a uniform weight if not given a specific weight for the initial portfolio
        """
        # initialize with the given weights
        if self.weights is not None:
            return self.weights
        # if no initializing or no weights for allocate method, return uniform weight
        if _weights is None:
            return self.uniform_weight()
        # return given weight
        return _weights
