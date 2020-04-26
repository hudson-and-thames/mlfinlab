# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class ExponentialGradient(OLPS):
    """
    This class implements the Exponential Gradient Portfolio strategy. It is reproduced with
    modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December YEAR), 33 pages. <https://arxiv.org/abs/1212.2129>.

    Exponential gradient strategy tracks the best performing stock in the last period while
    keeping previous portfolio information by using a regularization term.
    """

    def __init__(self, update_rule, eta=0.05):
        """
        Initializes with the designated update rule and eta, the learning rate.

        :param update_rule: (str) 'MU': Multiplicative Update, 'GP': Gradient Projection,
                                  'EM': Expectation Maximization.
        :param eta: (float) Learning rate.
        """
        super().__init__()
        self.eta = eta
        self.update_rule = update_rule

    def update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return new_weight: (np.array) New portfolio weights using exponential gradient.
        """
        # Get the window's price relative.
        relative_return = self.relative_return[time]

        # Take the dot product of weights and price relative.
        dot_product = np.dot(self.weights, relative_return)

        # Multiplicative update.
        if self.update_rule == 'MU':
            new_weight = self.weights * np.exp(self.eta * relative_return / dot_product)

        # Gradient projection.
        elif self.update_rule == 'GP':
            new_weight = self.weights + self.eta * (relative_return - np.ones(self.number_of_assets) * np.mean(relative_return) / dot_product)

        # Expectation maximization.
        elif self.update_rule == 'EM':
            new_weight = self.weights * (1 + self.eta * (relative_return / dot_product - 1))
        new_weight = self.normalize(new_weight)
        return new_weight
