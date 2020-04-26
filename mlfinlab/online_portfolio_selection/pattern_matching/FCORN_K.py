# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.univeral_portfolio import UniversalPortfolio
from mlfinlab.online_portfolio_selection.pattern_matching.FCORN import FCORN


class FCORN_K(UniversalPortfolio):
    """
    This class implements the Functional Correlation Driven Nonparametric Learning - top k experts strategy.
    """

    def __init__(self, k, window_values, rho_values, lambda_values):
        """
        Constructor.
        """
        self.k = k
        self.window_values = window_values
        self.rho_values = rho_values
        self.lambda_values = lambda_values
        self.number_of_experts = len(self.window_values) * len(self.rho_values) * len(self.lambda_values)
        super().__init__(number_of_experts=self.number_of_experts)

    def generate_experts(self):
        """
        Generates n experts for FCORN-K strategy

        :return:
        """
        self.expert_params = np.zeros((self.number_of_experts, 3))
        pointer = 0
        for _window in self.window_values:
            for _rho in self.rho_values:
                for _lambda in self.lambda_values:
                    self.expert_params[pointer] = [_window, _rho, _lambda]
                    pointer += 1

        for exp in range(self.number_of_experts):
            param = self.expert_params[exp]
            self.experts.append(FCORN(window=int(param[0]), rho=param[1], lamb=param[2]))


def main():
    """
    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    fcorn_k = FCORN_K(k=3, window_values=[2, 3, 4], rho_values=[.4, .6, .8], lambda_values=[10, 100, 500])
    fcorn_k.allocate(stock_price, resample_by='m')
    print(fcorn_k.all_weights)
    print(fcorn_k.portfolio_return)
    fcorn_k.portfolio_return.plot()


if __name__ == "__main__":
    main()
