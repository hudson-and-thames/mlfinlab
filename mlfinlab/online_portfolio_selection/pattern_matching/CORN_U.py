# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.UP import UP
from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN


class CORN_U(UP):
    """
    This class implements the Correlation Driven Nonparametric Learning - Uniform strategy.
    """

    # check -1 <= rho <= 1
    # check window >= 1
    def __init__(self, window_values, rho_values):
        """
        Constructor.
        """
        self.window_values = window_values
        self.rho_values = rho_values
        self.number_of_experts = len(self.window_values) * len(self.rho_values)
        super().__init__(number_of_experts=self.number_of_experts)

    def generate_experts(self):
        """
        Generates n experts for CORN-U strategy

        :return:
        """
        self.expert_params = np.zeros((self.number_of_experts, 2))
        pointer = 0
        for _window in self.window_values:
            for _rho in self.rho_values:
                self.expert_params[pointer] = [_window, _rho]
                pointer += 1

        for exp in range(self.number_of_experts):
            param = self.expert_params[exp]
            self.experts.append(CORN(int(param[0]), param[1]))

    def calculate_weights_on_experts(self):
        """
        Calculates the weight allocation on each experts
        Weights rebalanced to give equal allocation to all managers

        :return: (None) set weights_on_experts
        """

        # weight allocated is 1/n for all experts
        expert_returns_ratio = np.ones(self.expert_portfolio_returns.shape) / self.number_of_experts
        self.weights_on_experts = expert_returns_ratio


def main():
    """
    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn_u = CORN_U(window_values=[2, 3, 4], rho_values=[0.4, 0.6, 0.8])
    corn_u.allocate(stock_price, resample_by='m')
    print(corn_u.all_weights)
    print(corn_u.portfolio_return)
    corn_u.portfolio_return.plot()


if __name__ == "__main__":
    main()
