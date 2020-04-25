# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class ConstantRebalancedPortfolio(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from
    the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December YEAR), 33 pages. <https://arxiv.org/abs/1212.2129>.

    Constant Rebalanced Portfolio rebalances to a given weight each time period.
    """
    def __init__(self, weight=None):
        """
        Sets the recurring weights for the Constant Rebalanced Portfolio.

        :param weight: (list/np.array/pd.Dataframe) Initial weight set by the user.
        """
        super(ConstantRebalancedPortfolio, self).__init__()
        self.weight = weight

    def first_weight(self, weights):
        """
        Sets first weight for Constant Rebalanced Portfolio

        :param weights: (list/np.array/pd.Dataframe) initial weights set by the user.
        :return weights: (np.array) First portfolio weight.
        """
        # Initialize with given weights.
        if self.weight is not None:
            return self.weight
        # If no weights are given, automatically set weights to uniform weights.
        if weights is None:
            weights = self.uniform_weight()
        return weights
