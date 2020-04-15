# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class EG(OLPS):
    """
    Exponential Gradient
    """

    def __init__(self, eta=0.05, update_rule='MU'):
        """
        Constructor.
        """
        self.eta = eta
        self.update_rule = update_rule
        super().__init__()

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        past_relative_return = _relative_return[_time - 1]
        dot_product = np.dot(_weights, past_relative_return)

        if self.update_rule == 'MU':
            new_weight = _weights * np.exp(self.eta * past_relative_return / dot_product)
        elif self.update_rule == 'GP':
            new_weight = _weights + self.eta * (past_relative_return - np.sum(past_relative_return) / self.number_of_assets) / dot_product
        elif self.update_rule == 'EM':
            new_weight = _weights * (1 + self.eta * (past_relative_return/dot_product - 1))

        return self.normalize(new_weight)