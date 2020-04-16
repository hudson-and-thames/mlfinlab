# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
import cvxpy as cp

# General OLPS class
class OLPS(object):
    """
    Parent Class for all Online Portfolio Selection Algorithms
    This class broadly defines all variables and allocates a set of weights given a certain strategy
    """
    def __init__(self):
        """
        Initiating variables
        """
        # weights of the most recent time
        self.weights = None
        # all weights of the given time period
        self.all_weights = None
        # delayed portfolio in case we want only a later timeframe
        self.portfolio_start = None
        # asset names of the given data
        self.asset_name = None
        # number of assets in the given data
        self.number_of_assets = None
        # asset time index
        self.time = None
        # number of time periods over the given data
        self.number_of_time = None
        # final asset time index
        self.final_time = None
        # final number of time periods
        self.final_number_of_time = None
        # relative return data
        self.relative_return = None
        # final relative return data
        self.final_relative_return = None
        # portfolio return
        self.portfolio_return = None
        # pass dataframe on
        self.asset_prices = None

    # public method idea
    # OLPS.allocate(some_data)
    # OLPS.weights
    # OLPS.summary()
    # OLPS.returns
    def allocate(self,
                 asset_prices,
                 weights=None,
                 portfolio_start=0,
                 resample_by=None):
        """
        All OLPS algorithms can have their weight allocated using this method

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param portfolio_start: (int) delay the portfolio by n number of time
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """
        # Data Check method to make sure that the data fits the standards
        # asset name in right format
        # asset price in right format
        # resample_by in right format
        # weights add up to 1
        # resample function
        # not implemented yet
        self.check_asset(asset_prices, weights, portfolio_start, resample_by)

        # Data Prep to initialize all variables
        self.initialize(asset_prices, weights, portfolio_start, resample_by)

        # Actual weight calculation
        # For future portfolios only change __run() if we want to change it to batch style
        # or change update_weight to change stepwise function
        # or change first_weight if we delay our portfolio return start date
        self.run(weights, self.final_relative_return)

        # round weights and drop values that are less than the given threshold
        self.all_weights = self.round_weights(self.all_weights, threshold=1e-6)

        # Calculate Portfolio Returns based on all the weights calculated from the run method
        self.calculate_portfolio_returns(self.all_weights, self.final_relative_return)

        # convert everything to pd.Dataframe to make the information presentable
        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return)

    # check for valid dataset
    # raise ValueError
    def check_asset(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        """
        Checks if the given input value is valid

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param portfolio_start: (int) delay the portfolio by n number of time
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        :return: (None) raises ValueError if there are incorrect inputs
        """
        # is the dataset actually valid
        pass
        # check _asset_prices is dataframe
        # check weights size is _asset_prices column size
        # check that the weights sum to 1
        # _resample_by actually works - right parameter
        # _portfolio_start is a valid number and nonnegative

    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        """
        Initializes the important variables for the object

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param portfolio_start: (int) delay the portfolio by n number of time
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        :return: (None) Sets all the important information regarding the portfolio
        """
        # resample asset to either make the calculations shorter or to look at a longer time period
        if _resample_by is not None:
            _asset_prices = _asset_prices.resample(_resample_by).last()

        # set portfolio start
        # this information will be implemented later with more data on portfolio usage
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

        # calculate final relative return
        self.final_relative_return = self.calculate_relative_return(_asset_prices[self.portfolio_start:])

        # set final_weights
        self.all_weights = np.zeros((self.final_number_of_time, self.number_of_assets))

        # set portfolio_return
        self.portfolio_return = np.zeros((self.final_number_of_time, 1))

        # pass dataframe on in order to speed up process
        self.asset_prices = _asset_prices

    # for this one, it doesn't matter, but for subsequent complex selection problems, we might have to include a
    # separate run method for each iteration and not clog the allocate method.
    # after calculating the new weight add that to the all weights
    def run(self, _weights, _relative_return):
        """
        Runs the algorithm in a step by step fashion

        :param _weights: (list/np.array/pd.Dataframe) any initial weights that the user wants to use
        :param _relative_return: (np.array) the relative returns of the given price data
        :return: (None) Calculates all the weights given the range of time
        """
        # set initial weights
        self.weights = self.first_weight(_weights)
        # set initial all weights to be the first given weight
        self.all_weights[0] = self.weights

        # run the algorithm for the rest of data from time 1 to the end
        for t in range(1, self.final_number_of_time):
            # update weights
            self.weights = self.update_weight(self.weights, _relative_return, t)
            # set the next all_weights to be the changed weight
            self.all_weights[t] = self.weights

    # for this specific class, the weight doesn't change so just return the same weight
    # for other portfolios, most of the itmes we only have to change this for future iteration
    def update_weight(self, _weights, _relative_return, _time):
        """
        Updates portfolio weights for a given time, previous weight, and given relative return array

        :param _weights: (np.array) weight for the previous time period
        :param _relative_return:  (np.array) relative returns of a certain time period specified by the strategy
        :param _time: (int) time for the portfolio calculation
        :return _weights: (np.array) same weight as the input weight
        """
        # weight does not change
        return _weights

    def calculate_relative_return(self, _asset_prices):
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

    def calculate_rolling_correlation_coefficient(self, _relative_return):
        """
        Calculates the rolling correlation coefficient for a given relative return and window

        :param _relative_return: (np.array) relative returns of a certain time period specified by the strategy
        :return rolling_corr_coef: (np.array) rolling correlation coefficient over a given window
        """
        # take the log of the relative return
        # first calculate the rolling window the relative return
        # sum the data which returns the log of the window's return
        # take the exp to revert back to the original window's returns
        # calculate the correlation coefficient for the different window's overall returns
        rolling_corr_coef = np.corrcoef(np.exp(np.log(pd.DataFrame(_relative_return)).rolling(self.window).sum()))
        return rolling_corr_coef

    def calculate_rolling_moving_average(self, _asset_prices, _window, _reversion_method, _alpha):
        """
        Calculates the rolling moving average for Online Moving Average Reversion

        :param _asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param _window: (int) number of market windows
        :param _reversion_method: (int) number that represents the reversion method
        :param _alpha: (int) exponential weight for the second reversion method
        :return rolling_ma: (np.array) rolling moving average for the given reversion method
        """
        # MAR-1 reversion method: Simple Moving Average
        if _reversion_method == 1:
            rolling_ma = np.array(_asset_prices.rolling(_window).apply(lambda x: np.sum(x) / x[0] / _window))
        # Mar-2 reversion method: Exponential Moving Average
        elif _reversion_method == 2:
            rolling_ma = np.array(_asset_prices.ewm(alpha=_alpha, adjust=False).mean() / _asset_prices)
        return rolling_ma

    # initialize first weight
    # might change depending on algorithm
    def first_weight(self, _weights):
        """
        Returns the first weight of the given portfolio

        :param _weights: (list/np.array/pd.Dataframe) weights given by the user, none if not initialized
        :return _weights: (np.array) returns a uniform weight if not given a specific weight for the initial portfolio
        """
        # no weights given so return uniform weight
        if _weights is None:
            return self.uniform_weight(self.number_of_assets)
        # return given weight
        else:
            return _weights

    def uniform_weight(self, n):
        """
        Returns a uniform weight given n number assets

        :param n: (int) number of assets
        :return uni_weight: (np.array) uniform weights (1/n, 1/n, 1/n ...)
        """
        # divide by n after creating numpy arrays of one
        uni_weight = np.ones(n) / n
        return uni_weight

    def calculate_portfolio_returns(self, _all_weights, _relative_return):
        """
        Calculates portfolio returns

        :param _all_weights: (np.array) all weights of the given portfolio
        :param _relative_return: (np.array) relative returns of a certain time period specified by the strategy
        :return: (None) set current portfolio_return with the given value
        """
        # take the dot product of the relative returns and transpose of all_weights
        # diagonal of the resulting matrix will be the returns for each week
        # take the cumulative product of the returns to calculate returns over time
        self.portfolio_return = np.diagonal(np.dot(_relative_return, _all_weights.T)).cumprod()

    def normalize(self, _weights):
        """
        Normalize sum of weights to one

        :param _weights: (np.array) value of portfolio weights that has not been processed yet
        :return norm_weights: (np.array) divide by the total sum to create an array that sums to one
        """
        norm_weights = _weights / np.sum(_weights)
        return norm_weights

    def diag_mul(self, A, B):
        """
        Method to get a diagonal multiplication of two arrays
        --- Still in progress
        --- equivalent to np.diag(np.dot(A, B))

        :param A: (np.array)
        :param B: (np.array)
        :return diag_m: (np.array) diagonal of the dot product of two matrix
        """
        diag_m = (A * B.T).sum(-1)
        return diag_m

    def conversion(self, _all_weights, _portfolio_return):
        """
        Method to convert the given np.array to pd.Dataframe for visibility issues

        :param _all_weights: (np.array) portfolio weights calculated for all periods
        :param _portfolio_return: (np.array) cumulative portfolio returns for all periods
        :return: (None) set all_weights and portfolio_return to pd.Dataframe with corresponding index and column
        """
        self.all_weights = pd.DataFrame(_all_weights, index=self.final_time, columns=self.asset_name)
        self.portfolio_return = pd.DataFrame(_portfolio_return, index=self.final_time, columns=["Returns"])

    def round_weights(self, _all_weights, threshold=1e-6):
        """
        Method to drop weights below a certain threshold

        :param _all_weights: (np.array) portfolio weights calculated for all periods
        :param threshold: (float) drop all values below this threshold
        :return new_all_weights: (np.array) portfolio weights that dropped all values below threshold
        """
        # set all values below the threshold to 0
        new_all_weights = np.where(_all_weights < threshold, 0, _all_weights)
        # normalize the weights to sum of 1 in case the weights don't add up to 1
        new_all_weights = np.apply_along_axis(lambda x: x / np.sum(x), 1, new_all_weights)
        return new_all_weights

    def optimize(self, _optimize_array, _solver=None):
        """
        Method to return weights that maximize returns over a given _optimize_array

        :param _optimize_array: (np.array) relative returns for a given time period
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
        allocation_constraints = [
            cp.sum(weights) == 1,
            cp.min(weights) >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        # if there is a specified solver use it
        if _solver:
            problem.solve(solver=_solver)
        # if none specified, use default solver
        else:
            problem.solve()
        return weights.value

        # optimize the weight that minimizes the l2 norm
    def simplex_projection(self, _optimize_weight):
        """
        Method to calculate the simplex projection of the calculated weights

        :param _optimize_weight: (np.array) a weight that will be projected onto the simplex domain
        :return weights.value: (np.array) simplex projection of the original weight
        """
        # initialize weights for simplex projection
        weights = cp.Variable(self.number_of_assets)

        # taking the minimizing weights of l2 norm of the difference equates to simplex projection
        l2_norm = cp.norm(weights - _optimize_weight)

        # Optimization objective and constraints
        allocation_objective = cp.Minimize(l2_norm)
        allocation_constraints = [
                cp.sum(weights) == 1,
                weights >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
                objective=allocation_objective,
                constraints=allocation_constraints
        )
        problem.solve()
        return weights.value

    def sigmoid(self, x):
        """
        Generates the resulting sigmoid function

        :param x: (float) input for the sigmoid function
        :return sig: (float) sigmoid(x)
        """
        sig = 1 / (1 + np.exp(-x))
        return sig

    # Method to implement later for portfolio visualization

    # calculate the variance based on the price
    def volatility(self):
        pass

    # Calculate Sharpe Ratio
    def sharpe_ratio(self):
        # self.portfolio_sharpe_ratio = ((self.portfolio_return - risk_free_rate) / (self.portfolio_risk ** 0.5))
        pass

    # return maximum drawdown
    def maximum_drawdown(self):
        return 1 - min(self.portfolio_return)

    # return summary of the portfolio
    def summary(self):
        pass

def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    initial_portfolio = OLPS()
    initial_portfolio.allocate(stock_price)
    print(initial_portfolio.all_weights)
    print(initial_portfolio.portfolio_return)
    initial_portfolio.portfolio_return.plot()


if __name__ == "__main__":
    main()
