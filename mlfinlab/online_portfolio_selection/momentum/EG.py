# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class EG(OLPS):
    """
    This class implements the Exponential Gradient Portfolio strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    Exponential gradient strategy tracks the best performing stock in the last period while keeping previous portfolio
    information by using a regularization term
    """

    def __init__(self, eta=0.05, update_rule='MU'):
        """
        Initializes with eta and update_rule

        :param eta: (float) learning rate value
        :param update_rule: (str) 'MU': Multiplicative Update, 'GP': Gradient Projection, 'EM': Expectation Maximization
        """
        self.eta = eta
        self.update_rule = update_rule
        super().__init__()

    def update_weight(self, _weights, _relative_return, _time):
        """
        Updates weight given the update rule and eta, learning rate

        :param _weights: (np.array) portfolio weights of the previous time period
        :param _relative_return: (np.array) relative returns of the given time period
        :param _time: (int) time period
        :return new_weight: (np.array) new portfolio weights by using the designated update rule
        """
        past_relative_return = _relative_return[_time - 1]
        dot_product = np.dot(_weights, past_relative_return)

        if self.update_rule == 'MU':
            new_weight = _weights * np.exp(self.eta * past_relative_return / dot_product)
        elif self.update_rule == 'GP':
            new_weight = _weights + self.eta * (
                        past_relative_return - np.sum(past_relative_return) / self.number_of_assets) / dot_product
        elif self.update_rule == 'EM':
            new_weight = _weights * (1 + self.eta * (past_relative_return / dot_product - 1))
        new_weight = self.normalize(new_weight)
        return new_weight
