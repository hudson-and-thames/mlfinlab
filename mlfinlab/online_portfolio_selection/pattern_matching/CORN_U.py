# pylint: disable=missing-module-docstring
import cvxpy as cp
import numpy as np
import pandas as pd

from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN


class CORN_U(CORN):
    """
    This class implements the Correlation Driven Nonparametric Learning - Uniform strategy.
    """
    # check -1 <= rho <= 1
    # check window >= 1
    def __init__(self, number_of_window=3, number_of_rho=3):
        """
        Constructor.
        """
        self.number_of_window = number_of_window
        self.number_of_rho = number_of_rho
        self.experts = []
        self.expert_portfolio_returns = None
        self.expert_all_weights = None
        self.number_of_experts = None
        # this is the dataframe that we're looking at
        super().__init__()

    # totally changing the initialize function to incorporate calling all the objects
    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        # resample asset
        if _resample_by is not None:
            _asset_prices = _asset_prices.resample(_resample_by).last()

        # set portfolio start
        self.portfolio_start = _portfolio_start

        # set asset names
        self.asset_name = _asset_prices.columns

        # set time
        self.time = _asset_prices.index

        # calculate number of assets
        self.number_of_assets = self.asset_name.size

        # calculate number of time
        self.number_of_time = self.time.size

        # calculate relative returns and final relative returns
        self.relative_return = self.calculate_relative_return(_asset_prices)

        # set portfolio start
        self.portfolio_start = _portfolio_start

        # set final returns
        self.final_time = self.time[self.portfolio_start:]
        self.final_number_of_time = self.final_time.size

        # calcualte final relative return
        self.final_relative_return = self.calculate_relative_return(_asset_prices[self.portfolio_start:])

        # set final_weights
        self.all_weights = np.zeros((self.final_number_of_time, self.number_of_assets))

        # set portfolio_return
        self.portfolio_return = np.zeros((self.final_number_of_time, 1))

        # pass dataframe on
        self.asset_prices = _asset_prices

        # calculate total number of experts
        self.number_of_experts = self.number_of_window * self.number_of_rho
        
        # generate parameters for experts
        expert_param = self.generate_experts(self.number_of_window, self.number_of_rho)

        # generate all inividual CORN experts
        for exp in range(self.number_of_experts):
            self.experts.append(CORN(window=int(expert_param[exp][0]), rho=expert_param[exp][1]))

        # set experts portfolio returns and weights
        # 3d array a bit like a cube
        self.expert_portfolio_returns = np.zeros((self.final_number_of_time, self.number_of_experts))
        self.expert_all_weights = np.zeros((self.number_of_experts, self.final_number_of_time, self.number_of_assets))
            
    # run by iterating through the experts
    # could optimize for efficiency later on
    def run(self, _weights, _relative_return):
        # run allocate on all the experts
        for exp in range(self.number_of_experts):
            print(self.experts[exp].window, self.experts[exp].rho)
            # allocate to each experts
            self.experts[exp].allocate(self.asset_prices)
            # stack the weights
            self.expert_all_weights[exp] = self.experts[exp].all_weights
            # stack the portfolio returns
            self.expert_portfolio_returns[:, [exp]] = self.experts[exp].portfolio_return

        # uniform weight distribution for wealth between managers
        self.all_weights = np.mean(self.expert_all_weights, axis=0)

    def generate_experts(self, _number_of_window, _number_of_rho):
        """
        :param _length_of_window:
        :param _number_of_rho:
        :return:
        """
        # calculate total number of experts
        total_number = _number_of_window * _number_of_rho
        # np array format of first value as window, second as rho
        experts_param = np.zeros((total_number, 2))
        pointer = 0
        # making windows size of 0 to n-1
        for window in range(1, _number_of_window + 1):
            # making rho size of 0 to (rho - 1)/rho
            for rho in range(_number_of_rho):
                experts_param[pointer] = [window, rho / _number_of_rho]
                pointer += 1
        return experts_param
        # experts_param[0] is the first expert's param


def main():
    """
    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn_u = CORN_U(number_of_window=20, number_of_rho=10)
    corn_u.allocate(stock_price, resample_by='m')
    print(corn_u.all_weights)
    print(corn_u.portfolio_return)
    corn_u.portfolio_return.plot()


if __name__ == "__main__":
    main()
