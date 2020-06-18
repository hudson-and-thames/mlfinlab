# pylint: disable=missing-module-docstring
import sys
import pandas as pd
import numpy as np
import cvxpy as cp


class OLPS:
    """
    Online Portfolio Selection is an algorithmic trading strategy that sequentially allocates
    capital among a group of assets to maximize the final returns of the investment.

    Traditional theories for portfolio selection, such as Markowitz’s Modern Portfolio Theory,
    optimize the balance between the portfolio's risks and returns. However, OLPS is founded on
    the capital growth theory, which solely focuses on maximizing the returns of the current
    portfolio.

    Through these walkthroughs of different portfolio selection strategies, we hope to introduce
    a set of different selection tools available for everyone. Most of the works will be based on
    Dr. Bin Li and Dr. Steven Hoi’s book, Online Portfolio Selection: Principles and Algorithms,
    and further recent papers will be implemented to assist the development and understanding of
    these unique portfolio selection strategies.

    OLPS is the parent class for all resulting Online Portfolio Selection Strategies.

    This class broadly defines all variables and allocates a set of weights for a certain strategy.

    Upon weights allocation the possible outputs are:

    - ``self.weights`` (np.array) Final portfolio weights prediction.

    - ``self.all_weights`` (pd.DataFrame) Portfolio weights for the time period.

    - ``self.asset_name`` (list) Name of assets.

    - ``self.number_of_assets`` (int) Number of assets.

    - ``self.time`` (datetime) Time index of the given data.

    - ``self.length_of_time`` (int) Number of time periods.

    - ``self.relative_return`` (np.array) Relative returns of the assets.

    - ``self.portfolio_return`` (pd.DataFrame) Cumulative portfolio returns over time.

    - ``self.asset_prices`` (pd.DataFrame) Historical asset prices (daily close).
    """

    def __init__(self):
        self.weights = None  # (np.array) Final portfolio weights prediction.
        self.all_weights = None  # (pd.DataFrame) Portfolio weights for the time period.
        self.asset_name = None  # (list) Name of assets.
        self.number_of_assets = None  # (int) Number of assets.
        self.time = None  # (datetime) Time index of the given data.
        self.length_of_time = None  # (int) Number of time periods.
        self.relative_return = None  # (np.array) Relative returns of the assets.
        self.portfolio_return = None  # (pd.DataFrame) Cumulative portfolio returns over time.
        self.asset_prices = None  # (pd.DataFrame) Historical asset prices (daily close).

    def allocate(self, asset_prices, weights=None, resample_by=None, verbose=False):
        """
        Allocates weight according to a set of update rules.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user. If None, weights
                                                     will default to uniform weights.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        :param verbose: (bool) Prints progress bar if true.
        """
        # Check to ensure inputs are correct.
        self._check_asset(asset_prices, weights)

        # Initialize all variables.
        self._initialize(asset_prices, weights, resample_by)

        # Iterate through data and calculate weights.
        self._run(weights, verbose)

        # Round weights and drop values that are less than the given threshold.
        self._round_weights(threshold=1e-6)

        # Calculate portfolio returns based on weights calculated from the run method.
        self._calculate_portfolio_returns(self.all_weights, self.relative_return)

        # Convert everything to dataframe to make the information presentable.
        self._conversion()

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        """
        # Resample asset.
        if resample_by is not None:
            asset_prices = asset_prices.resample(resample_by).last()

        # Set asset names.
        self.asset_name = asset_prices.columns

        # Set time.
        self.time = asset_prices.index

        # Calculate number of assets.
        self.number_of_assets = self.asset_name.size

        # Calculate number of time.
        self.length_of_time = self.time.size

        # Calculate relative returns and final relative returns.
        self.relative_return = self._calculate_relative_return(asset_prices)

        # Set initial weights.
        self.weights = weights

        # Set all_weights.
        self.all_weights = np.zeros((self.length_of_time + 1, self.number_of_assets))

        # Set portfolio returns.
        self.portfolio_return = np.zeros((self.length_of_time, 1))

        # Pass dataframe to speed up process for universal portfolio.
        self.asset_prices = asset_prices

    def _run(self, weights, verbose):
        """
        Runs the algorithm by iterating through the given data.

        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param verbose: (bool) Prints progress bar if true.
        """
        # Set initial weights.
        self.weights = self._first_weight(weights)

        # Set initial_all weights to be the first given weight.
        self.all_weights[0] = self.weights

        # Run the algorithm for the rest of data from time 1.
        for time in range(self.length_of_time):
            # Update weights.
            self.weights = self._update_weight(time)
            self.all_weights[time + 1] = self.weights
            # Print progress bar.
            if verbose:
                self._print_progress(time + 1, prefix='Progress:', suffix='Complete')

        # Remove final prediction as that information is stored in self.weights.
        self.all_weights = self.all_weights[:-1]

    def _first_weight(self, weights):
        """
        Returns the first weight of the given portfolio. If the first weight is not given, initialize weights to
        uniform weights.

        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :return: (np.array) First portfolio weight.
        """
        # If no weights are given, return uniform weights.
        if weights is None:
            weights = self._uniform_weight()
        return weights

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Identical weights are returned for OLPS.
        """
        # Weights do not change for this class.
        new_weights = self.all_weights[time]
        return new_weights

    def _calculate_portfolio_returns(self, all_weights, relative_return):
        """
        Calculates cumulative portfolio returns.

        :param all_weights: (np.array) Portfolio weights for the time period.
        :param relative_return: (np.array) Relative returns of the assets.
        """
        # Take the dot product of the relative returns and transpose of all_weights.
        # Diagonal of the resulting matrix will be the returns for each week.
        # Calculate returns by taking the cumulative product.
        self.portfolio_return = np.diagonal(np.dot(relative_return, all_weights.T)).cumprod()

    def _conversion(self):
        """
        Converts the given np.array to pd.Dataframe.
        """
        # Convert all_weights.
        self.all_weights = pd.DataFrame(self.all_weights, index=self.time, columns=self.asset_name)

        # Convert portfolio_return.
        self.portfolio_return = pd.DataFrame(self.portfolio_return, index=self.time, columns=["Returns"])

    def _optimize(self, optimize_array, solver=cp.SCS):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) Relative returns of the assets for a given time period.
        :param solver: (cp.solver) Solver for cvxpy
        :return: (np.array) Weights that maximize the returns for the given array.
        """

        # Initialize weights for the optimization problem.
        weights = cp.Variable(self.number_of_assets)

        # Use cp.log and cp.sum to make the cost function a convex function.
        # Multiplying continuous returns equates to summing over the log returns.
        portfolio_return = cp.sum(cp.log(optimize_array @ weights))

        # Optimization objective and constraints.
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [cp.sum(weights) == 1, cp.min(weights) >= 0]

        # Define and solve the problem.
        problem = cp.Problem(objective=allocation_objective, constraints=allocation_constraints)

        # Solve and return the resulting weights.
        problem.solve(warm_start=True, solver=solver)
        return weights.value

    def _round_weights(self, threshold=1e-6):
        """
        Drops weights that are below a certain threshold.

        :param threshold: (float) Drop all values below this threshold.
        """
        # Set all values below the threshold to 0.
        new_all_weights = np.where(self.all_weights < threshold, 0, self.all_weights)

        # Adjust weights to have the weights sum to 1.
        new_all_weights = np.apply_along_axis(lambda x: x / np.sum(x), 1, new_all_weights)
        self.all_weights = new_all_weights

    def _uniform_weight(self):
        """
        Returns a uniform weight of assets.

        :return: (np.array) Uniform weights (1/n, 1/n, 1/n ...).
        """
        # Divide by number of assets after creating numpy arrays of one.
        uni_weight = np.ones(self.number_of_assets) / self.number_of_assets
        return uni_weight

    def _print_progress(self, iteration, prefix='', suffix='', decimals=1, bar_length=50):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

        :param iteration: (int) Current iteration.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """
        str_format = "{0:." + str(decimals) + "f}"
        # Calculate the percent completed.
        percents = str_format.format(100 * (iteration / float(self.length_of_time)))
        # Calculate the length of bar.
        filled_length = int(round(bar_length * iteration / float(self.length_of_time)))
        # Fill the bar.
        block = '█' * filled_length + '-' * (bar_length - filled_length)
        # Print new line.
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, block, percents, '%', suffix)),

        if iteration == self.length_of_time:
            sys.stdout.write('\n')
        sys.stdout.flush()

    @staticmethod
    def _normalize(weights):
        """
        Normalize sum of weights to one.

        :param weights: (np.array) Pre-processed weights that have not been normalized yet.
        :return: (np.array) Adjusted weights that sum to 1.
        """
        norm_weights = weights / np.sum(weights)
        return norm_weights

    @staticmethod
    def _calculate_relative_return(asset_prices):
        """
        Calculates the relative return of a given price data.

        :param asset_prices: (pd.DataFrame) Dataframe of historical asset prices.
        :return: (np.array) Relative returns of the assets.
        """
        # First calculate the percent change of each time period.
        # First row is NaN because there is no initial change, so we replace that row with 0.
        # Add 1 to all values to replicate relative returns for each week.
        # Change type to np.array
        relative_return = np.array(asset_prices.pct_change().fillna(0) + 1)
        return relative_return

    @staticmethod
    def _check_asset(asset_prices, weights):
        """
        Checks if the given input values are valid.

        :param asset_prices: (pd.DataFrame) Dataframe of historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        """
        # If weights have been given by the user.
        if weights is not None:
            # Check if the number of assets match.
            if len(weights) != asset_prices.shape[1]:
                raise ValueError("Given portfolio weights do not match data shape.")
            # Check if the weights sum to 1.
            np.testing.assert_almost_equal(np.sum(weights), 1)

        # Check if given data is in dataframe format.
        if not isinstance(asset_prices, pd.DataFrame):
            raise ValueError("Asset prices matrix must be a dataframe. Please change the data.")

        # Check if index of dataframe is indexed by date.
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date. Please parse dates "
                             "and set the index as dates.")

        # Check that the given data has no null value.
        if asset_prices.isnull().any().sum() != 0:
            raise ValueError("The given dataframe contains values of null. Please remove the null "
                             "values.")

        # Check that the given data has no values of 0.
        if (asset_prices == 0).any().sum() != 0:
            raise ValueError("The given dataset contains values of 0. Please remove the 0 values.")

    @staticmethod
    def _simplex_projection(weight):
        """
        Calculates the simplex projection of weights.
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

        :param weight: (np.array) Weight to be projected onto the simplex domain.
        :return: (np.array) Simplex projection of the original weight.
        """
        # Sort in descending order.
        _mu = np.sort(weight)[::-1]

        # Calculate adjusted sum.
        adjusted_sum = np.cumsum(_mu) - 1
        j = np.arange(len(weight)) + 1

        # Determine the conditions.
        cond = _mu - adjusted_sum / j > 0

        # If all conditions are false, return uniform weight.
        if not cond.any():
            uniform_weight = np.ones(len(weight)) / len(weight)
            return uniform_weight

        # Define max rho.
        rho = float(j[cond][-1])

        # Define theta.
        theta = adjusted_sum[cond][-1] / rho

        # Calculate new weight.
        new_weight = np.maximum(weight - theta, 0)
        return new_weight
