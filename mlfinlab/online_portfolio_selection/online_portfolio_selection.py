# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
import cvxpy as cp


class OLPS:
    """
    Online Portfolio Selection is an algorithmic trading strategy that sequentially allocates capital among a group of
    assets to maximize the final returns of the investment.

    Traditional theories for portfolio selection, such as Markowitz’s Modern Portfolio Theory, optimize the balance
    between the portfolio's risks and returns. However, OLPS is founded on the capital growth theory, which solely
    focuses on maximizing the returns of the current portfolio.

    Through these walkthroughs of different portfolio selection strategies, we hope to introduce a set of different
    selection tools available for everyone. Most of the works will be based on Dr. Bin Li and Dr. Steven Hoi’s book,
    Online Portfolio Selection: Principles and Algorithms, and further recent papers will be implemented to assist the
    development and understanding of these unique portfolio selection strategies.

    OLPS is the parent class for all resulting Online Portfolio Selection Strategies.

    This class broadly defines all variables and allocates a set of weights given a certain strategy.
    """

    def __init__(self):
        self.weights = None  # (np.array) final portfolio weights prediction
        self.all_weights = None  # (pd.DataFrame) portfolio weights for the time period
        self.asset_name = None  # (list) name of assets
        self.number_of_assets = None  # (int) number of assets
        self.time = None  # (datetime) time index of the given data
        self.length_of_time = None  # (int) number of time periods
        self.relative_return = None  # (np.array) relative returns of the assets
        self.portfolio_return = None  # (pd.DataFrame) cumulative portfolio returns over time
        self.asset_prices = None  # (pd.DataFrame) a dataframe of historical asset prices (daily close)

    def allocate(self, asset_prices, weights=None, resample_by=None):
        """
        Allocates weight according to a set of update rules.

        :param asset_prices: (pd.DataFrame) dataframe of historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) initial weights set by the user.
        :param resample_by: (str) specifies how to resample the prices.
        """
        # checks to ensure inputs are correct
        self.check_asset(asset_prices, weights)

        # initializes all variables
        self.initialize(asset_prices, weights, resample_by)

        # iterates through data and calculates weights
        self.run(weights)

        # round weights and drop values that are less than the given threshold
        self.round_weights(threshold=1e-6)

        # calculate portfolio returns based on weights calculated from the run method
        self.calculate_portfolio_returns(self.all_weights, self.relative_return)

        # converts everything to pd.Dataframe to make the information presentable
        self.conversion(self.all_weights, self.portfolio_return)

    def initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) dataframe of historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) initial weights set by the user.
        :param resample_by: (str) specifies how to resample the prices.
        """
        # resample asset
        if resample_by is not None:
            asset_prices = asset_prices.resample(resample_by).last()

        # set asset names
        self.asset_name = asset_prices.columns

        # set time
        self.time = asset_prices.index

        # calculate number of assets
        self.number_of_assets = self.asset_name.size

        # calculate number of time
        self.length_of_time = self.time.size

        # calculate relative returns and final relative returns
        self.relative_return = self.calculate_relative_return(asset_prices)

        # set initial weights
        self.weights = weights

        # set all_weights, last weight is the predicted weight for the next time period
        self.all_weights = np.zeros((self.length_of_time + 1, self.number_of_assets))

        # set portfolio_return
        self.portfolio_return = np.zeros((self.length_of_time, 1))

        # pass dataframe on to speed up process for universal portfolio
        self.asset_prices = asset_prices

    def run(self, weights):
        """
        Runs the algorithm by iterating through the given data.

        :param weights: (list/np.array/pd.Dataframe) initial weights set by the user.
        """
        # set initial weights
        self.weights = self.first_weight(weights)

        # set initial_all weights to be the first given weight
        self.all_weights[0] = self.weights

        # run the algorithm for the rest of data from time 1
        for time in range(self.length_of_time):
            # update weights
            self.weights = self.update_weight(time)
            self.all_weights[time + 1] = self.weights

        # remove final prediction because that information is stored in self.weights
        self.all_weights = self.all_weights[:-1]

    def first_weight(self, weights):
        """
        Returns the first weight of the given portfolio. If the first weight is not given, initialize weights to
        uniform weights.

        :param weights: (list/np.array/pd.Dataframe) initial weights set by the user.
        :return (weights): (np.array) returns the first portfolio weight.
        """
        # if no weights are given, return uniform weights
        if weights is None:
            weights = self.uniform_weight()
        return weights

    def update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) current time period.
        """
        # weights do not change for this class
        return self.all_weights[time]

    def calculate_portfolio_returns(self, all_weights, relative_return):
        """
        Calculates cumulative portfolio returns.

        :param all_weights: (np.array) portfolio weights for the time period.
        :param relative_return: (np.array) relative returns of the assets.
        """
        # take the dot product of the relative returns and transpose of all_weights[:-1]
        # diagonal of the resulting matrix will be the returns for each week
        # take the cumulative product of the returns to calculate returns over time
        self.portfolio_return = np.diagonal(np.dot(relative_return, all_weights.T)).cumprod()

    def conversion(self, all_weights, portfolio_return):
        """
        Converts the given np.array to pd.Dataframe.

        :param all_weights: (np.array) portfolio weights for the time period.
        :param portfolio_return: (np.array) cumulative portfolio returns for all periods.
        """
        # converts all_weights
        self.all_weights = pd.DataFrame(all_weights, index=self.time, columns=self.asset_name)

        # converts portfolio_return
        self.portfolio_return = pd.DataFrame(portfolio_return, index=self.time, columns=["Returns"])

    def optimize(self, optimize_array, solver=cp.SCS):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) relative returns of the assets for a given time period.
        :param solver: (cp.solver) set the solver to be a particular cvxpy solver.
        :return weights.value: (np.array) weights that maximize the returns for the given optimize_array.
        """

        # initialize weights for optimization problem
        weights = cp.Variable(self.number_of_assets)

        # use cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        portfolio_return = cp.sum(cp.log(optimize_array * weights))

        # optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [cp.sum(weights) == 1, cp.min(weights) >= 0]

        # define and solve the problem
        problem = cp.Problem(objective=allocation_objective, constraints=allocation_constraints)

        # solve and return the resulting weights
        problem.solve(warm_start=True, solver=solver)
        return weights.value

    def round_weights(self, threshold=1e-6):
        """
        Drops weights that are below a certain threshold.

        :param threshold: (float) drop all values below this threshold.
        """
        # set all values below the threshold to 0
        new_all_weights = np.where(self.all_weights < threshold, 0, self.all_weights)

        # adjust weights to have the weights sum to 1
        new_all_weights = np.apply_along_axis(lambda x: x / np.sum(x), 1, new_all_weights)
        self.all_weights = new_all_weights

    def uniform_weight(self):
        """
        Returns a uniform weight of assets.

        :return uni_weight: (np.array) uniform weights (1/n, 1/n, 1/n ...).
        """
        # divide by number of assets after creating numpy arrays of one
        uni_weight = np.ones(self.number_of_assets) / self.number_of_assets
        return uni_weight

    @staticmethod
    def normalize(weights):
        """
        Normalize sum of weights to one.

        :param weights: (np.array) pre-processed weights that have not been normalized yet.
        :return norm_weights: (np.array) adjusted weights that sum to 1.
        """
        norm_weights = weights / np.sum(weights)
        return norm_weights

    @staticmethod
    def calculate_relative_return(asset_prices):
        """
        Calculates the relative return of a given price data.

        :param asset_prices: (pd.DataFrame) dataframe of historical asset prices.
        :return relative_return: (np.array) relative returns of the assets.
        """
        # first calculate the percent change of each time period
        # first row is nan because there is no initial change, so we replace the row with 0
        # add 1 to all values to replicate relative returns for each week
        # change type to np.array
        relative_return = np.array(asset_prices.pct_change().fillna(0) + 1)
        return relative_return

    @staticmethod
    def check_asset(asset_prices, weights):
        """
        Checks if the given input values are valid.

        :param asset_prices: (pd.DataFrame) dataframe of historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) initial weights set by the user.
        """
        # if weights are given
        if weights is not None:
            # check if number of assets match
            if len(weights) != asset_prices.shape[1]:
                raise ValueError("Given portfolio weights do not match data shape")
            # check if weights sum to 1
            np.testing.assert_almost_equal(np.sum(weights), 1)

        # check if given data is in dataframe format
        if not isinstance(asset_prices, pd.DataFrame):
            raise ValueError("Asset prices matrix must be a dataframe")

        # check if index of dataframe is indexed by date
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date.")

    @staticmethod
    def simplex_projection(weight):
        """
        Calculates the simplex projection of the weights
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

        :param weight: (np.array) calculated weight to be projected onto the simplex domain
        :return weights.value: (np.array) simplex projection of the original weight
        """
        # return itself if already a simplex projection
        if np.sum(weight) == 1 and np.all(weight >= 0):
            return weight

        # sort descending
        _mu = np.sort(weight)[::-1]

        # adjusted sum
        adjusted_sum = np.cumsum(_mu) - 1

        # number
        j = np.arange(len(weight)) + 1

        # condition
        cond = _mu - adjusted_sum / j > 0

        # define max rho
        rho = float(j[cond][-1])

        # define theta
        theta = adjusted_sum[cond][-1] / rho

        # calculate new weight
        new_weight = np.maximum(weight - theta, 0)
        return new_weight
