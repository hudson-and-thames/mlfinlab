# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS
from mlfinlab.online_portfolio_selection.benchmarks.constant_rebalanced_portfolio import ConstantRebalancedPortfolio


class UniversalPortfolio(OLPS):
    """
    This class implements the Best Constant Rebalanced Portfolio strategy. It is reproduced with
    modification from the following paper:
    Cover, T.M. (1991), Universal Portfolios. Mathematical Finance, 1: 1-29.
    <http://www-isl.stanford.edu/~cover/papers/portfolios_side_info.pdf>.

    Universal Portfolio is
    """
    def __init__(self, number_of_experts):
        """
        Constructor.
        """
        self.experts = []  # Array to store all experts
        self.number_of_experts = number_of_experts  # Set the number of experts.
        self.expert_params = None  # Each expert's parameters.
        self.expert_portfolio_returns = None  # All experts' portfolio returns over time.
        self.expert_all_weights = None  # Each experts' weights over time.
        self.weights_on_experts = None  # Capital allocation on each experts.
        super(UniversalPortfolio, self).__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices.
        """
        # Initialize the same variables as OLPS.
        super(UniversalPortfolio, self)._initialize(asset_prices, weights, resample_by)

        # Generate all experts.
        self.generate_experts()

        # Set experts portfolio returns and weights.
        self.expert_portfolio_returns = np.zeros((self.length_of_time, self.number_of_experts))
        self.expert_all_weights = np.zeros((self.number_of_experts, self.length_of_time,
                                            self.number_of_assets))
        self.weights = np.zeros((self.number_of_experts, self.number_of_assets))

    def _run(self, weights):
        """
        Runs the algorithm by iterating through the given data.

        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        """
        # Run allocate on all the experts.
        for exp in range(self.number_of_experts):
            # Allocate to each experts.
            self.experts[exp].allocate(self.asset_prices)
            # Stack the weights.
            self.expert_all_weights[exp] = self.experts[exp].all_weights
            # Stack the portfolio returns.
            self.expert_portfolio_returns[:, [exp]] = self.experts[exp].portfolio_return
            # Stack final weights.
            self.weights[exp] = self.experts[exp].weights

        self.calculate_weights_on_experts()
        # Uniform weight distribution for wealth between managers.
        self.calculate_all_weights()

    def generate_experts(self):
        """
        Generate experts with the specified parameter.
        """
        # Generate randomized weights within the simplex.
        self.generate_simplex(self.number_of_experts, self.number_of_assets)
        # Universal Portfolio looks at different CRP weights.
        for exp in range(self.number_of_experts):
            self.experts.append(ConstantRebalancedPortfolio(weight=self.expert_params[exp]))

    def generate_simplex(self, number_of_experts, number_of_assets):
        """
        Generate uniform points on a simplex domain.
        https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

        :param number_of_experts: (int) Number of experts.
        :param number_of_assets: (int) Number of assets.
        """
        # Create a randomized array with dimensions of (number_of_experts, number_of_assets - 1).
        simplex = np.sort(np.random.random((number_of_experts, number_of_assets - 1)))

        # Stack a column of zeros on the left, and stack a column of ones on the right.
        # Calculate the difference of each interval.
        # Each interval equates to a uniform sampling of the simplex domain.
        simplex = np.diff(np.hstack([np.zeros((number_of_experts, 1)), simplex,
                                     np.ones((number_of_experts, 1))]))
        self.expert_params = simplex

    def calculate_weights_on_experts(self):
        """
        Calculates the weight allocation on each experts.
        The initial weights do not change, but total weights do change as underlying asset
        prices fluctuate.
        """
        # Calculate each expert's cumulative return ratio for each time period.
        expert_returns_ratio = np.apply_along_axis(lambda x: x/np.sum(x), 1,
                                                   self.expert_portfolio_returns[:-1])
        print(expert_returns_ratio.shape)
        # Initial weights are evenly distributed among all experts.
        expert_returns_ratio = np.vstack((self.uniform_experts(),
                                          expert_returns_ratio))
        self.weights_on_experts = expert_returns_ratio

    def uniform_experts(self):
        """
        Returns a uniform weight of experts.

        :return uni_weight: (np.array) Uniform weights (1/n, 1/n, 1/n ...).
        """
        # Divide by number of assets after creating numpy arrays of one.
        uni_weight = np.ones(self.number_of_experts) / self.number_of_experts
        return uni_weight

    def calculate_all_weights(self):
        """
        Universal Portfolio allocates the same weight to all experts.
        The weights will be adjusted each week due to market fluctuations.
        """

        # Calculate the product of the distribution matrix with the 3d experts x all weights matrix.
        # https://stackoverflow.com/questions/58588378/
        # how-to-matrix-multiply-a-2d-numpy-array-with-a-3d-array-to-give-a-3d-array
        d_shape = self.weights_on_experts.shape[:1] + self.expert_all_weights.shape[1:]
        weight_change = (self.weights_on_experts @ self.expert_all_weights.reshape(
                self.expert_all_weights.shape[0], -1)).reshape(d_shape)
        # We are looking at the diagonal cross section of the multiplication.
        self.all_weights = np.diagonal(weight_change, axis1=0, axis2=1).T
