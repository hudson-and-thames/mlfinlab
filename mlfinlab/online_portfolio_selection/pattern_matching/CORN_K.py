# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN
from mlfinlab.online_portfolio_selection.UP import UP


class CORN_K(UP):
    """
    This class implements the Correlation Driven Nonparametric Learning - top k experts strategy.
    """
    # check -1 <= rho <= 1
    # check window >= 1
    def __init__(self, k, window_values, rho_values):
        """
        Constructor.
        """
        self.k = k
        self.window_values = window_values
        self.rho_values = rho_values
        self.number_of_experts = len(self.window_values) * len(rho_values)
        super().__init__(number_of_experts=self.number_of_experts)

    def generate_experts(self):
        """
        Generates n experts for CORN-K strategy

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

        # wealth is not uniformaly distributed only the top k experts get 1/k of the wealth
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # indexs of top k for each time
        top_k = np.apply_along_axis(lambda x: np.argpartition(x, -self.k)[-self.k:], 1, self.expert_portfolio_returns)
        # pop last row off
        top_k = top_k[:-1]
        # create a wealth distribution matrix
        top_k_distribution = np.zeros(self.expert_portfolio_returns.shape)
        # first weight is uniform
        top_k_distribution[0] = self.uniform_weight(self.number_of_experts)
        # for each week put the multiplier for each expert
        # new array where each row represents that week's allocation to the k experts
        for time in range(1, top_k.shape[0] + 1):
            top_k_distribution[time][top_k[time - 1]] = 1/self.k

        self.weights_on_experts = top_k_distribution



def main():
    """
    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn_k = CORN_K(k=3, window_values=[2,3,4], rho_values=[0.4,0.6,0.8])
    corn_k.allocate(stock_price, resample_by='m')
    print(corn_k.all_weights)
    print(corn_k.portfolio_return)
    corn_k.portfolio_return.plot()


if __name__ == "__main__":
    main()
