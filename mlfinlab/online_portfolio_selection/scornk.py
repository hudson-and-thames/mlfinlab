# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.cornk import CORNK
from mlfinlab.online_portfolio_selection.scorn import SCORN


class SCORNK(CORNK):
    """
    This class implements the Symmetric Correlation Driven Nonparametric Learning - K strategy. It
    is reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Symmetric Correlation Driven Nonparametric Learning - K formulates a number of SCORN experts and
    tracks the experts performance over time. Each period, the strategy decides to allocate capital
    to the top-k experts until the previous time period. This strategy takes an ensemble approach to
    the top performing experts.
    """

    pass
