# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS


class EG(OLPS):
    """
    This class implements the Exponential Gradient Portfolio strategy. It is reproduced with
    modification from the following paper:
    `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

    Exponential gradient strategy tracks the best performing stock in the last period while
    keeping previous portfolio information by using a regularization term.
    """

    def __init__(self, update_rule, eta=0.05):
        """
        Initializes with the designated update rule and eta, the learning rate.

        :param update_rule: (str) 'MU': Multiplicative Update, 'GP': Gradient Projection,
                                  'EM': Expectation Maximization. All three update rules return
                                  similar results with slight differences.
        :param eta: (float) Learning rate with range of [0, inf). Low rate indicates the passiveness
                            of following the momentum and high rate indicates the aggressivness of
                            following the momentum. Depending on the dataset either 0.05 or 20 usually
                            have the highest returns.
        """

        pass

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) New portfolio weights using exponential gradient.
        """

        pass
