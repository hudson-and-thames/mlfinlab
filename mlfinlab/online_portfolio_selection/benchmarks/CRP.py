# pylint: disable=missing-module-docstring
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class CRP(OLPS):
    """
    This class implements the Constant Rebalanced Portfolio strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    Constant Rebalanced Portfolio rebalances to a given weight for each time period.
    """