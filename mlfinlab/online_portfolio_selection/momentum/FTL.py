# pylint: disable=missing-module-docstring
import pandas as pd
from mlfinlab.online_portfolio_selection import BCRP


class FTL(BCRP):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        return self.optimize(_relative_return[:_time])