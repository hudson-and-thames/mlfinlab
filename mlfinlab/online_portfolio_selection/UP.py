# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
from mlfinlab.online_portfolio_selection.OLPS import OLPS
from mlfinlab.online_portfolio_selection.benchmarks.CRP import CRP


class UP(OLPS):
    """
    This class implements the Universal Portfolio Strategy
    """

    def __init__(self, number_of_experts):
        """
        Constructor.
        """
        # array to "store" all the experts
        self.experts = []
        # set the number of experts
        self.number_of_experts = number_of_experts
        # set each expert's parameter
        self.expert_params = None
        # np.array of all expert's portfolio returns over time
        self.expert_portfolio_returns = None
        # 3d np.array of each expert's weights over time
        self.expert_all_weights = None
        super(UP, self).__init__()

    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        # initialize the same variables as OLPS
        super(UP, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)

        # generate all the different weights for each experts
        self.expert_params = self.generate_simplex(self.number_of_experts, self.number_of_assets)

        # generate all experts
        self.generate_experts(self.number_of_experts)

        # set experts portfolio returns and weights
        self.expert_portfolio_returns = np.zeros((self.final_number_of_time, self.number_of_experts))
        self.expert_all_weights = np.zeros((self.number_of_experts, self.final_number_of_time, self.number_of_assets))

    def generate_simplex(self, _number_of_experts, _number_of_assets):
        """
        Method to generate uniform points on a simplex domain
        https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

        :param _number_of_portfolio: (int) number of portfolios that the universal portfolio wants to create
        :param _number_of_assets: (int) number of assets
        :return simplex.T: (np.array) random simplex points
        """
        # first create a randomized array with number of portfolios and number of assets minus one
        simplex = np.sort(np.random.random((_number_of_experts, _number_of_assets - 1)))

        # stack a column of zeros on the left
        # stack a column of ones on the right
        # take the difference of each interval which equates to a uniform sampling of the simplex domain
        simplex = np.diff(np.hstack([np.zeros((_number_of_experts, 1)), simplex, np.ones((_number_of_experts, 1))]))
        return simplex

    def generate_experts(self, _number_of_experts):
        """
        Generate the experts with the specified parameter
        Can easily swap out for different generations for different UP algorithms

        :param _number_of_experts: (int) number of experts generated for universal portfolio
        :return: (None) Initialize each strategies
        """
        for exp in range(self.number_of_experts):
            self.experts.append(CRP())

    def run(self, _weights, _relative_return):
        """
        Runs all experts by iterating through the initiated array

        :param _weights:
        :param _relative_return:
        :return:
        """
        # run allocate on all the experts
        for exp in range(self.number_of_experts):
            # allocate to each experts
            self.experts[exp].allocate(self.asset_prices, weights=self.expert_params[exp])
            # stack the weights
            self.expert_all_weights[exp] = self.experts[exp].all_weights
            # stack the portfolio returns
            self.expert_portfolio_returns[:, [exp]] = self.experts[exp].portfolio_return

        # uniform weight distribution for wealth between managers
        self.calculate_all_weights(self.expert_all_weights, self.expert_portfolio_returns)

    def calculate_all_weights(self, expert_all_weights, expert_portfolio_returns):
        """
        UP allocates the same weight to all experts
        The weights will be adjusted each week due to market fluctuations

        :param expert_all_weights: (np.array) 3d array
        :param expert_portfolio_returns (np.array) 2d array
        :return average_weights: (np.array) 2d array
        """
        # create new np.array with final all weights
        new_all_weights = np.zeros((self.final_number_of_time, self.number_of_assets))
        # calculate each expert's cumulative return ratio for each time period
        expert_returns_ratio = np.apply_along_axis(lambda x: x/np.sum(x), 1, expert_portfolio_returns[:-1])
        expert_returns_ratio = np.vstack((self.uniform_weight(self.number_of_experts), expert_returns_ratio))
        # subsequent weights will be calculated by changes in experts' portfolio returns
        print(expert_returns_ratio)
        # calculate the product of the distribution matrix with the 3d experts x all weights matrix
        # https://stackoverflow.com/questions/58588378/how-to-matrix-multiply-a-2d-numpy-array-with-a-3d-array-to-give-a-3d-array
        d_shape = expert_returns_ratio.shape[:1] + expert_all_weights.shape[1:]
        weight_change = (expert_returns_ratio @ expert_all_weights.reshape(expert_all_weights.shape[0], -1)).reshape(d_shape)
        # we are looking at the diagonal cross section of the multiplication
        self.all_weights = np.diagonal(weight_change, axis1=0, axis2=1).T


def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    universal_portfolio = UP(1000)
    universal_portfolio.allocate(stock_price)
    print(universal_portfolio.all_weights)
    print(universal_portfolio.portfolio_return)
    universal_portfolio.portfolio_return.plot()


if __name__ == "__main__":
    main()
