# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS
from mlfinlab.online_portfolio_selection.benchmarks.constant_rebalanced_portfolio import ConstantRebalancedPortfolio


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
        # wealth weights on each expert
        self.weights_on_experts = None
        super(UP, self).__init__()

    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        # initialize the same variables as OLPS
        super(UP, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)

        # generate all experts
        self.generate_experts()

        # set experts portfolio returns and weights
        self.expert_portfolio_returns = np.zeros((self.final_number_of_time, self.number_of_experts))
        self.expert_all_weights = np.zeros((self.number_of_experts, self.final_number_of_time, self.number_of_assets))

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

        self.calculate_weights_on_experts()
        # uniform weight distribution for wealth between managers
        self.calculate_all_weights()

    def generate_experts(self):
        """
        Generate the experts with the specified parameter
        Can easily swap out for different generations for different UP algorithms

        :return: (None) Initialize each strategies
        """
        self.generate_simplex(self.number_of_experts, self.number_of_assets)
        for exp in range(self.number_of_experts):
            self.experts.append(ConstantRebalancedPortfolio(weights=self.expert_params[exp]))

    def generate_simplex(self, _number_of_experts, _number_of_assets):
        """
        Method to generate uniform points on a simplex domain
        https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

        :param _number_of_experts: (int) number of experts that the universal portfolio wants to create
        :param _number_of_assets: (int) number of assets
        :return: (None) set expert_params as the weights
        """
        # first create a randomized array with number of portfolios and number of assets minus one
        simplex = np.sort(np.random.random((_number_of_experts, _number_of_assets - 1)))

        # stack a column of zeros on the left
        # stack a column of ones on the right
        # take the difference of each interval which equates to a uniform sampling of the simplex domain
        simplex = np.diff(np.hstack([np.zeros((_number_of_experts, 1)), simplex, np.ones((_number_of_experts, 1))]))
        self.expert_params = simplex

    def calculate_weights_on_experts(self):
        """
        Calculates the weight allocation on each experts
        The initial weights don't change, but total weights do change as underlying asset price fluctuates

        :return: (None) set weights_on_experts
        """
        # calculate each expert's cumulative return ratio for each time period
        expert_returns_ratio = np.apply_along_axis(lambda x: x/np.sum(x), 1, self.expert_portfolio_returns[:-1])
        # initial weights is evenly distributed among all experts
        expert_returns_ratio = np.vstack((self.uniform_weight(self.number_of_experts), expert_returns_ratio))
        self.weights_on_experts = expert_returns_ratio

    def calculate_all_weights(self):
        """
        UP allocates the same weight to all experts
        The weights will be adjusted each week due to market fluctuations

        :return average_weights: (np.array) 2d array
        """

        # calculate the product of the distribution matrix with the 3d experts x all weights matrix
        # https://stackoverflow.com/questions/58588378/how-to-matrix-multiply-a-2d-numpy-array-with-a-3d-array-to-give-a-3d-array
        d_shape = self.weights_on_experts.shape[:1] + self.expert_all_weights.shape[1:]
        weight_change = (self.weights_on_experts @ self.expert_all_weights.reshape(self.expert_all_weights.shape[0], -1)).reshape(d_shape)
        # we are looking at the diagonal cross section of the multiplication
        self.all_weights = np.diagonal(weight_change, axis1=0, axis2=1).T
