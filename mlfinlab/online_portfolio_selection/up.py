# pylint: disable=missing-module-docstring
# pylint: disable=too-many-instance-attributes
import sys
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS
from mlfinlab.online_portfolio_selection.crp import CRP


class UP(OLPS):
    """
    This class implements the Universal Portfolio strategy. It is reproduced with
    modification from the following paper:
    `Cover, T.M. (1991), Universal Portfolios. Mathematical Finance, 1: 1-29.
    <http://www-isl.stanford.edu/~cover/papers/portfolios_side_info.pdf>`_

    Universal Portfolio acts as a fund of funds, generating a number of experts with unique
    strategies. Cover's original universal portfolio integrates over the total simplex domain,
    but because it is not possible for us to calculate all possibilties, we generate a random
    distribution of points.

    The allocation method to each experts can be changed. If no
    allocation method is given, Universal Portfolio will not rebalance among the experts. Other
    allocation methods include uniform allocation among experts and top-k experts, which allocate
    capital based on the top-k performing experts until the last period.
    """
    def __init__(self, number_of_experts, weighted='hist_performance', k=1):
        """
        Initializes Universal Portfolio with the given number of experts, method of capital
        allocation to each experts, and k-value for Top-K experts.

        :param number_of_experts: (int) Number of total experts.
        :param weighted: (str) Capital allocation method. 'hist_performance': Historical Performance,
                               'uniform': Uniform Weights, 'top-k': Top-K experts.
        :param k: (int) Number of experts to choose your portfolio. Only necessary if weighted is
                        'top-k'. Typically lower values of k are more optimal for higher returns.
        """
        self.experts = []  # (list) Array to store all experts
        self.number_of_experts = number_of_experts  # (int) Set the number of experts.
        self.expert_params = None  # (np.array) Each expert's parameters.
        self.expert_portfolio_returns = None  # (np.array) All experts' portfolio returns over time.
        self.expert_all_weights = None  # (np.array) Each experts' weights over time.
        self.expert_weights = None  # (np.array) Each experts' final portfolio weights
        self.weights_on_experts = None  # (np.array) Capital allocation on each experts.
        self.weighted = weighted  # (np.array) Weights allocated to each experts.
        self.k = k  # (int) Number of top-k experts.
        super(UP, self).__init__()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        """
        # Initialize the same variables as OLPS.
        super(UP, self)._initialize(asset_prices, weights, resample_by)

        # Generate all experts.
        self._generate_experts()

        # Set experts' portfolio returns.
        self.expert_portfolio_returns = np.zeros((self.length_of_time, self.number_of_experts))
        # Set all experts' weights
        self.expert_all_weights = np.zeros((self.number_of_experts, self.length_of_time,
                                            self.number_of_assets))
        # Set all experts' predicted weights.
        self.expert_weights = np.zeros((self.number_of_experts, self.number_of_assets))

    def _run(self, weights, verbose):
        """
        Runs the algorithm by iterating through the given data.

        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param verbose: (bool) Prints progress bar if true.
        """
        # Run allocate on all the experts.
        for exp in range(self.number_of_experts):
            # Allocate to each experts.
            self.experts[exp].allocate(self.asset_prices)
            # Stack the weights.
            self.expert_all_weights[exp] = self.experts[exp].all_weights
            # Stack the portfolio returns.
            self.expert_portfolio_returns[:, [exp]] = self.experts[exp].portfolio_return
            # Stack predicted weights.
            self.expert_weights[exp] = self.experts[exp].weights
            # Print progress bar.
            if verbose:
                self._print_progress(exp + 1, prefix='Progress:', suffix='Complete')
        # Calculate capital allocation on each experts.
        self._calculate_weights_on_experts()
        # Uniform weight distribution for wealth between managers.
        self._calculate_all_weights()

    def _calculate_weights_on_experts(self):
        """
        Calculates the weight allocation on each experts.
        'hist_performance': Historical Performance.
        'uniform': Uniform Weights.
        'top-k': Top-K experts.
        """
        # If capital allocation is based on historical performances.
        if self.weighted == 'hist_performance':
            # Calculate each expert's cumulative return ratio for each time period.
            expert_returns_ratio = np.apply_along_axis(lambda x: x/np.sum(x), 1,
                                                       self.expert_portfolio_returns)
            # Initial weights are evenly distributed among all experts.
            expert_returns_ratio = np.vstack((self._uniform_experts(), expert_returns_ratio))
            self.weights_on_experts = expert_returns_ratio

        # If capital allocation is based on uniform weights.
        elif self.weighted == 'uniform':
            # Equal allocation.
            uniform_ratio = np.ones(
                self.expert_portfolio_returns.shape) / self.number_of_experts
            # Initial weights are evenly distributed among all experts.
            uniform_ratio = np.vstack((self._uniform_experts(), uniform_ratio))
            self.weights_on_experts = uniform_ratio

        # If capital allocation is based on top-K experts.
        elif self.weighted == 'top-k':
            # Only the top k experts get 1/k of the wealth.
            # `<https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array>`_
            # Get the indices of top k experts for each time.
            top_k = np.apply_along_axis(lambda x: np.argpartition(x, -self.k)[-self.k:], 1,
                                        self.expert_portfolio_returns)
            # Create a wealth distribution matrix.
            top_k_distribution = np.zeros(self.expert_portfolio_returns.shape)
            # For each week set the multiplier for each expert.
            # Each row represents the week's allocation to the k experts.

            for time in range(top_k.shape[0]):
                top_k_distribution[time][top_k[time]] = 1 / self.k
            # Initial weights are evenly distributed among all experts.
            top_k_distribution = np.vstack((self._uniform_experts(), top_k_distribution))
            self.weights_on_experts = top_k_distribution
        else:
            raise ValueError("Please put in 'hist_performance' for Historical Performance, "
                             "'uniform' for Uniform Distribution, or 'top-k' for top-K experts.")

    def recalculate_k(self, k):
        """
        Calculates the existing strategy with a different k value. The user does not have to
        rerun the entire strategy, but can simply recalculate with another k parameter.

        :param k: (int) Number of new top-k experts.
        """
        # Check that k value is an integer.
        if not isinstance(k, int):
            raise ValueError("K value must be an integer.")

        # Check that k value is at least 1.
        if k < 1:
            raise ValueError("K value must be greater than or equal to 1.")

        # Check that k value is less than window * rho.
        if k > self.number_of_experts:
            raise ValueError("K must be less than or equal to window * rho.")

        self.k = k

        # Calculate capital allocation on each experts.
        self._calculate_weights_on_experts()

        # Uniform weight distribution for wealth between managers.
        self._calculate_all_weights()

        # Round weights and drop values that are less than the given threshold.
        self._round_weights(threshold=1e-6)

        # Calculate portfolio returns based on weights calculated from the run method.
        self._calculate_portfolio_returns(self.all_weights, self.relative_return)

        # Convert everything to dataframe to make the information presentable.
        self._conversion()

    def _uniform_experts(self):
        """
        Returns a uniform weight of experts.

        :return: (np.array) Uniform weights (1/n, 1/n, 1/n ...).
        """
        # Divide by number of assets after creating numpy arrays of one.
        uni_weight = np.ones(self.number_of_experts) / self.number_of_experts
        return uni_weight

    def _calculate_all_weights(self):
        # pylint: disable=unsubscriptable-object
        """
        Calculate portfolio's overall weights and final predicted weights with information from
        each expert's weights.
        """

        # Calculate the product of the distribution matrix with the 3d experts x all weights matrix.
        # `<https://stackoverflow.com/questions/58588378/how-to-matrix-multiply-a-2d-numpy-array-with-a-3d-array-to-give-a-3d-array>`_
        d_shape = self.weights_on_experts[:-1].shape[:1] + self.expert_all_weights.shape[1:]
        reshaped_all_weights = self.expert_all_weights.reshape(self.expert_all_weights.shape[0], -1)
        weight_change = np.dot(self.weights_on_experts[:-1], reshaped_all_weights).reshape(d_shape)
        # We are looking at the diagonal cross section of the multiplication.
        self.all_weights = np.diagonal(weight_change, axis1=0, axis2=1).T

        # Calculate final predicted weights.
        self.weights = np.dot(self.weights_on_experts[[-1]], self.expert_weights)

    def _generate_experts(self):
        """
        Generate experts with the specified parameter.
        """
        # Generate randomized weights within the simplex.
        self._generate_simplex(self.number_of_experts, self.number_of_assets)
        # Universal Portfolio looks at different CRP weights.
        for exp in range(self.number_of_experts):
            self.experts.append(CRP(weight=self.expert_params[exp]))

    def _generate_simplex(self, number_of_experts, number_of_assets):
        """
        Generate uniform points on a simplex domain.
        `<https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex>`_

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
        # Set the parameters as the randomly generated simplex weights.
        self.expert_params = simplex

    def _print_progress(self, iteration, prefix='', suffix='', decimals=1, bar_length=50):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        `<https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a>`_

        :param iteration: (int) Current iteration.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """
        str_format = "{0:." + str(decimals) + "f}"
        # Calculate the percent completed.
        percents = str_format.format(100 * (iteration / float(self.number_of_experts)))
        # Calculate the length of bar.
        filled_length = int(round(bar_length * iteration / float(self.number_of_experts)))
        # Fill the bar.
        block = '█' * filled_length + '-' * (bar_length - filled_length)
        # Print new line.
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, block, percents, '%', suffix)),

        if iteration == self.number_of_experts:
            sys.stdout.write('\n')
        sys.stdout.flush()
