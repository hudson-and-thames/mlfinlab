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

    This class broadly defines all variables and allocates a set of weights given a certain strategy

    :ivar weights: (np.array) final portfolio weights prediction
    :ivar all_weights: (pd.DataFrame) portfolio weights for the time period
    :ivar asset_name: (list) name of assets
    :ivar number_of_assets: (int) number of assets
    :ivar time: (datetime) time index of the given data
    :ivar length_of_time: (int) number of time periods
    :ivar relative_return: (np.array) relative returns of the assets
    :ivar portfolio_return: (pd.DataFrame) cumulative portfolio returns over time
    :ivar asset_prices: (pd.DataFrame) a dataframe of historical asset prices (daily close)
    """

    def __init__(self):
        self.weights = None
        self.all_weights = None
        self.asset_name = None
        self.number_of_assets = None
        self.time = None
        self.length_of_time = None
        self.relative_return = None
        self.portfolio_return = None
        self.asset_prices = None

    def allocate(self,
                 asset_prices,
                 weights=None,
                 resample_by=None):
        """
        Allocates weight according to a set of update rules

        :param asset_prices: (pd.DataFrame) a dataframe of historical asset prices (daily close)
        :param weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """
        # checks to ensure inputs are correct
        check_asset(asset_prices, weights, resample_by)

        # initializes all variables
        self.initialize(asset_prices, weights, resample_by)

        # iterates through data to and calculates weights
        self.run(weights, self.relative_return)

        # round weights and drop values that are less than the given threshold
        self.round_weights(threshold=1e-6)

        # calculate portfolio returns based on weights calculated from the run method
        self.calculate_portfolio_returns(self.all_weights, self.relative_return)

        # converts everything to pd.Dataframe to make the information presentable
        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return)

    def initialize(self,
                   _asset_prices,
                   _weights,
                   _resample_by):
        """
        Initializes the important variables for the object

        :param _asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param _weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param _resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        :return: (None) Sets all the important information regarding the portfolio
        """
        # resample asset
        if _resample_by is not None:
            _asset_prices = _asset_prices.resample(_resample_by).last()

        # set asset names
        self.asset_name = _asset_prices.columns

        # set time
        self.time = _asset_prices.index

        # calculate number of assets
        self.number_of_assets = self.asset_name.size

        # calculate number of time
        self.length_of_time = self.time.size

        # calculate relative returns and final relative returns
        self.relative_return = calculate_relative_return(_asset_prices)

        # set all_weights, last weights is the predicted weight for the next time period
        self.all_weights = np.zeros((self.length_of_time+1, self.number_of_assets))

        # set portfolio_return
        self.portfolio_return = np.zeros((self.length_of_time, 1))

        # pass dataframe on in order to speed up process
        self.asset_prices = _asset_prices

    def run(self,
            _weights,
            _relative_return):
        """
        Runs the algorithm in a step by step method to iterate through the given data

        :param _weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param _relative_return: (np.array) relative returns of the assets
        :return: (None) sets all_weights for the given data
        """
        # set initial weights
        self.weights = self.first_weight(_weights)
        # set initial all weights to be the first given weight
        self.all_weights[0] = self.weights

        # run the algorithm for the rest of data from time 1
        for time in range(self.length_of_time):
            # update weights
            self.weights = self.update_weight(time)
            self.all_weights[time+1] = self.weights

        # remove final prediction because that information is stored in self.weights
        self.all_weights = self.all_weights[:-1]

    def first_weight(self,
                     _weights):
        """
        Returns the first weight of the given portfolio
        Initializes to uniform weight if not given a certain weight

        :param _weights: (list/np.array/pd.Dataframe) weights given by the user, or none if not initialized
        :return (_weights): (np.array) first portfolio weights
        """
        # if no weights are given, return uniform weight
        if _weights is None:
            _weights = self.uniform_weight()
        # return given weight
        return _weights

    def update_weight(self,
                      _time):
        """
        Updates portfolio weights

        :param _time: (int) current time period
        :return (None) sets new weights to be the same as old weights
        """
        # weight do not change for this class
        return self.weights

    def calculate_portfolio_returns(self, _all_weights, _relative_return):
        """
        Calculates cumulative portfolio returns

        :param _all_weights: (np.array) portfolio weights for the time period
        :param _relative_return: (np.array) relative returns of the assets
        :return: (None) set portfolio_return as cumulative portfolio returns over time
        """
        # take the dot product of the relative returns and transpose of all_weights[:-1]
        # diagonal of the resulting matrix will be the returns for each week
        # take the cumulative product of the returns to calculate returns over time
        self.portfolio_return = np.diagonal(np.dot(_relative_return, _all_weights.T)).cumprod()

    def conversion(self, _all_weights, _portfolio_return):
        """
        Converts the given np.array to pd.Dataframe for visibility

        :param _all_weights: (np.array) portfolio weights for the time period
        :param _portfolio_return: (np.array) cumulative portfolio returns for all periods
        :return: (None) set all_weights and portfolio_return to pd.Dataframe
        """
        self.all_weights = pd.DataFrame(_all_weights, index=self.time, columns=self.asset_name)
        self.portfolio_return = pd.DataFrame(_portfolio_return, index=self.time, columns=["Returns"])

    def optimize(self,
                 _optimize_array,
                 _solver=None):
        """
        Calculates weights that maximize returns over a given _optimize_array

        :param _optimize_array: (np.array) relative returns of the assets for a given time period
        :param _solver: (cp.SOLVER) set the solver to be a particular cvxpy solver
        :return weights.value: (np.array) weights that maximize the returns for the given optimize_array
        """
        # calcualte length of time
        length_of_time = _optimize_array.shape[0]
        # calculate number of assets
        number_of_assets = _optimize_array.shape[1]
        # edge case to speed up calculation
        if length_of_time == 1:
            # in case that the optimize array is only one row, weights will be 1 for the highest relative return asset
            best_idx = np.argmax(_optimize_array)
            # initialize np.array of zeros
            weight = np.zeros(number_of_assets)
            # add 1 to the best performing stock
            weight[best_idx] = 1
            return weight

        # initialize weights for optimization problem
        weights = cp.Variable(self.number_of_assets)

        # used cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        portfolio_return = cp.sum(cp.log(_optimize_array * weights))

        # Optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [cp.sum(weights) == 1, cp.min(weights) >= 0]
        # Define and solve the problem
        problem = cp.Problem(objective=allocation_objective, constraints=allocation_constraints)
        # if there is a specified solver use it
        if _solver:
            problem.solve(warm_start=True, solver=_solver)
        else:
            problem.solve(warm_start=True)
        return weights.value

    def round_weights(self,
                      threshold=1e-6):
        """
        Drops weights below a certain threshold

        :param _all_weights: (np.array) portfolio weights for the time period
        :param threshold: (float) drop all values below this threshold
        :return (None): (np.array) sets all_weights as cleaned portfolio weights for the time period
        """
        # set all values below the threshold to 0
        new_all_weights = np.where(self.all_weights < threshold, 0, self.all_weights)
        # normalize the weights to sum of 1 in case the weights don't add up to 1
        new_all_weights = np.apply_along_axis(lambda x: x / np.sum(x), 1, new_all_weights)
        self.all_weights = new_all_weights

    def normalize(self,
                  _weights):
        """
        Normalize sum of weights to one

        :param _weights: (np.array) value of portfolio weights that has not been processed yet
        :return (None): (np.array) normalize the input and sets self.weights
        """
        norm_weights = _weights / np.sum(_weights)
        self.weights = norm_weights

    def uniform_weight(self):
        """
        Returns a uniform weight of assets

        :return uni_weight: (np.array) uniform weights (1/n, 1/n, 1/n ...)
        """
        # divide by n after creating numpy arrays of one
        uni_weight = np.ones(self.number_of_assets) / self.number_of_assets
        return uni_weight


def calculate_relative_return(_asset_prices):
    """
    Calculates the relative return of a given price data

    :param _asset_prices: (pd.Dataframe/np.array) historical price of the given assets
    :return relative_return: (np.array) relative returns of a certain time period specified by the strategy
    """
    # first calculate the percent change of each time period
    # first row is nan because there is no initial change, so we replce that value with 0
    # add 1 to all values to replicate relative returns for each week
    # change type to np.array
    relative_return = np.array(_asset_prices.pct_change().fillna(0) + 1)
    return relative_return


def sigmoid(val):
    """
    Generates the resulting sigmoid function

    :param val: (float) input for the sigmoid function
    :return sig: (float) sigmoid(x)
    """
    res = 1 / (1 + np.exp(-val))
    return res


def simplex_projection(_optimize_weight):
    """
    Calculates the simplex projection of the weights
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    :param _optimize_weight: (np.array) a weight that will be projected onto the simplex domain
    :return weights.value: (np.array) simplex projection of the original weight
    """

    # return itself if already a simplex projection
    if np.sum(_optimize_weight) == 1 and np.all(_optimize_weight >= 0):
        return _optimize_weight

    # sort descending
    _mu = np.sort(_optimize_weight)[::-1]

    # adjusted sum
    adjusted_sum = np.cumsum(_mu) - 1

    # number
    j = np.arange(len(_optimize_weight)) + 1

    # condition
    cond = _mu - adjusted_sum / j > 0

    # define max rho
    rho = float(j[cond][-1])

    # define theta
    theta = adjusted_sum[cond][-1] / rho

    # calculate new weight
    new_weight = np.maximum(_optimize_weight - theta, 0)
    return new_weight


def check_asset(_asset_prices,
                _weights,
                _resample_by):
    """
    Checks if the given input value is valid

    :param _asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
    :param _weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
    :param _resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                              None for no resampling
    :return: (None) raises ValueError if there are incorrect inputs
    """

    # Check if _asset_prices were given
    if _asset_prices is None:
        raise ValueError("You need to supply price returns data")

    # if weights are given
    if _weights is not None:
        # check if number of assets match
        if len(_weights) != _asset_prices.shape[1]:
            raise ValueError("Given portfolio weights do not match data shape")
        # check if weights sum to 1
        np.testing.assert_almost_equal(np.sum(_weights), 1)

    # check if given data is in dataframe format
    if not isinstance(_asset_prices, pd.DataFrame):
        raise ValueError("Asset prices matrix must be a dataframe")

    # check if index of dataframe is indexed by date
    if not isinstance(_asset_prices.index, pd.DatetimeIndex):
        raise ValueError("Asset prices dataframe must be indexed by date.")
