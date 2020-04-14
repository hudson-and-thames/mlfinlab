# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.pattern_matching.CORN import CORN
from mlfinlab.online_portfolio_selection.pattern_matching.CORN_U import CORN_U


class CORN_K(CORN_U):
    """
    This class implements the Correlation Driven Nonparametric Learning - Uniform strategy.
    """
    # check -1 <= rho <= 1
    # check window >= 1
    def __init__(self, k=5, window=10,rho=10):
        """
        Constructor.
        """
        self.k = k
        self.window = window
        self.rho = rho
        super().__init__()

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
        # calculate the product of the distribution matrix with the 3d expers x all weights matrix
        # https://stackoverflow.com/questions/58588378/how-to-matrix-multiply-a-2d-numpy-array-with-a-3d-array-to-give-a-3d-array
        d_shape = top_k_distribution.shape[:1] + self.expert_all_weights.shape[1:]
        weight_change = (top_k_distribution @ self.expert_all_weights.reshape(self.expert_all_weights.shape[0], -1)).reshape(d_shape)
        # we are looking at the diagonal cross section of the multiplication
        self.all_weights = np.diagonal(weight_change, axis1=0, axis2=1).T


def main():
    """
    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn_k = CORN_K(k=5)
    corn_k.allocate(stock_price, resample_by='m')
    print(corn_k.all_weights)
    print(corn_k.portfolio_return)
    corn_k.portfolio_return.plot()


if __name__ == "__main__":
    main()
