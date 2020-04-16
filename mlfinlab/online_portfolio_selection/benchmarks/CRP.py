# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class CRP(OLPS):
    """
    This class implements the Constant Rebalanced Portfolio strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    Constant Rebalanced Portfolio rebalances to a given weight for each time period.
    """
    def __init__(self, weights):
        super(CRP, self).__init__()
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
        elif _weights is None:
            return self.uniform_weight(self.number_of_assets)
        # return given weight
        else:
            return _weights