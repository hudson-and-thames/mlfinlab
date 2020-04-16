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
    def __init__(self, number_of_window=3, number_of_rho=3):
        """
        Constructor.
        """
        self.number_of_window = number_of_window
        self.number_of_rho = number_of_rho
        self.number_of_experts = number_of_window * number_of_rho
        super().__init__(number_of_experts=self.number_of_experts)

    def generate_experts(self):
        """
        :param _length_of_window:
        :param _number_of_rho:
        :return:
        """
        self.expert_params = np.zeros((self.number_of_experts, 2))
        pointer = 0
        for _window in range(1, self.number_of_window + 1):
            for _rho in range(1, self.number_of_rho + 1):
                self.expert_params[pointer] = [_window, _rho / self.number_of_rho]
                pointer += 1

        for exp in range(self.number_of_experts):
            param = self.expert_params[exp]
            self.experts.append(CORN(int(param[0]), param[1]))

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
            self.experts[exp].allocate(self.asset_prices)
            # stack the weights
            self.expert_all_weights[exp] = self.experts[exp].all_weights
            # stack the portfolio returns
            self.expert_portfolio_returns[:, [exp]] = self.experts[exp].portfolio_return

        # uniform weight distribution for wealth between managers
        self.calculate_all_weights(self.expert_all_weights, self.expert_portfolio_returns)

    def calculate_all_weights(self, expert_all_weights, expert_portfolio_returns):
        """
        CORN-U allocates the same weight to all experts in every time period

        :param expert_all_weights: (np.array) 3d array
        :param expert_portfolio_returns (np.array) 2d array
        :return average_weights: (np.array) 2d array
        """
        # weight allocated is 1/n for all experts
        expert_returns_ratio = np.ones(expert_portfolio_returns.shape) / self.number_of_experts

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
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn_u = CORN_U(number_of_window=20, number_of_rho=20)
    corn_u.allocate(stock_price, resample_by='m')
    print(corn_u.all_weights)
    print(corn_u.portfolio_return)
    corn_u.portfolio_return.plot()


if __name__ == "__main__":
    main()
