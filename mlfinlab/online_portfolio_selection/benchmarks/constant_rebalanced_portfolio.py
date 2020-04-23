# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class ConstantRebalancedPortfolio(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. <https://arxiv.org/abs/1212.2129>.

    Constant Rebalanced Portfolio rebalances to a given weight for each time period.

    :ivar weights: (np.array) any initial weights that the user wants to use
    """
    def __init__(self, weights=None):
        """
        Given weights will set the recurring weights for the Constant Rebalanced Portfolio.

        :param weights: (list/np.array/pd.Dataframe) weights given by the user, none if not initialized
        """
        super(ConstantRebalancedPortfolio, self).__init__()
        self.weights = weights

    def first_weight(self, _weights):
        """
        Sets the recurring weight of the Constant Rebalanced Portfolio by changing the first_weight method.

        :param _weights: (list/np.array/pd.Dataframe) weights given by the user, none if not initialized
        :return _weights: (np.array) returns a uniform weight if not given a specific weight for the initial portfolio
        """
        # initialize with the given weights
        if self.weights is not None:
            return self.weights
        # if no weights are given, automatically set weights to uniform weights
        if _weights is None:
            _weights = self.uniform_weight()
        # return given weight
        return _weights
