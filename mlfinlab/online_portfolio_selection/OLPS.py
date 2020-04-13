# pylint: disable=missing-module-docstring
import pandas as pd
import numpy as np
import cvxpy as cp


# General OLPS class
class OLPS(object):
    # Initialize
    def __init__(self):
        """
        :param weights: (pd.DataFrame) final weight of portfolio
        :param all_weights: (pd.DataFrame) all weights of portfolio
        :param portfolio_return: (pd.DataFrame) all returns of portfolio
        """
        # weights
        self.weights = None
        self.all_weights = None
        # delayed portfolio
        self.portfolio_start = None
        # asset names
        self.asset_name = None
        self.number_of_assets = None
        # asset time
        self.time = None
        self.number_of_time = None
        # final asset time
        self.final_time = None
        self.final_number_of_time = None
        # relative return and final relative return
        self.relative_return = None
        self.final_relative_return = None
        # portfolio return
        self.portfolio_return = None

        # self.asset_prices = None
        # self.covariance_matrix = None
        # self.portfolio_risk = None
        # self.portfolio_sharpe_ratio = None
        # self.expected_returns = None
        # self.returns_estimator = ReturnsEstimation()

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
        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param weights: any weights
        :param portfolio_start: (int) delay the portfolio by n number of time
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """
        # Data Check
        # Some sort of initial check to make sure data fits the standards
        # asset name in right format
        # asset price in right format
        # resample_by in right format
        # weights add up to 1
        # resample function
        # not implemented yet
        self.check_asset(asset_prices, weights, portfolio_start, resample_by)

        # Data Prep
        self.initialize(asset_prices, weights, portfolio_start, resample_by)

        # Actual weight calculation
        # For future portfolios only change __run() if we want to change it to batch style
        # or change __update_weight to change stepwise function
        # or change __first_weight if we delay our portfolio return start date
        self.run(weights, self.final_relative_return)

        # round weights
        self.all_weights = self.round_weights(self.all_weights)

        # Calculate Portfolio Returns
        self.calculate_portfolio_returns(self.all_weights, self.final_relative_return)

        # convert everything to make presentable
        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return)

    # check for valid dataset
    # raise ValueError
    def check_asset(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        # is the dataset actually valid
        pass
        # check _asset_prices is dataframe
        # check weights size is _asset_prices column size
        # _resample_by actually works
        # _portfolio_start is a valid number

    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        # resample asset
        if _resample_by is not None:
            _asset_prices = _asset_prices.resample(_resample_by).last()

        # set portfolio start
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

        # calcualte final relative return
        self.final_relative_return = self.calculate_relative_return(_asset_prices[self.portfolio_start:])

        # set final_weights
        self.all_weights = np.zeros((self.final_number_of_time, self.number_of_assets))

        # set portfolio_return
        self.portfolio_return = np.zeros((self.final_number_of_time, self.number_of_assets))

    # for this one, it doesn't matter, but for subsequent complex selection problems, we might have to include a
    # separate run method for each iteration and not clog the allocate method.
    # after calculating the new weight add that to the all weights
    def run(self, _weights, _relative_return):
        # set initial weights
        self.weights = self.first_weight(_weights)
        self.all_weights[0] = self.weights

        # Run the Algorithm for the rest of data
        for t in range(1, self.final_number_of_time):
            # update weights
            self.weights = self.update_weight(self.weights, _relative_return, t)
            self.all_weights[t] = self.weights

    # for the first one, just return the same weight
    # only have to change this for future iteration
    def update_weight(self, _weights, _relative_return, _time):
        return _weights

    # calculate relative returns
    def calculate_relative_return(self, _asset_prices):
        # percent change of each row
        # first row is blank because no change, so make it 0
        # add 1 to all values so that the values can be multiplied easily
        # change to numpy array
        relative_return = np.array(_asset_prices.pct_change().fillna(0) + 1)
        return relative_return

    # calculate rolling correlation coefficient
    def calculate_rolling_correlation_coefficient(self, _relative_return):
        rolling_corr_coef = np.corrcoef(np.exp(np.log(pd.DataFrame(_relative_return)).rolling(self.window).sum()))
        return rolling_corr_coef

    # calculate rolling moving average for OLMAR
    def calculate_rolling_moving_average(self, _asset_prices, _window, _reversion_method, _alpha):
        if _reversion_method == 1:
            return np.array(_asset_prices.rolling(_window).apply(lambda x: np.sum(x) / x[0] / _window))
        elif _reversion_method == 2:
            return np.array(_asset_prices.ewm(alpha=_alpha, adjust=False).mean() / _asset_prices)

    # initialize first weight
    # might change depending on algorithm
    def first_weight(self, _weights):
        if _weights is None:
            return self.uniform_weight(self.number_of_assets)
        else:
            return _weights

    # return uniform weights numpy array (1/n, 1/n, 1/n ...)
    def uniform_weight(self, n):
        return np.ones(n) / n

    # calculate portfolio returns
    def calculate_portfolio_returns(self, _all_weights, _relative_return):
        self.portfolio_return = np.diagonal(np.dot(_relative_return, _all_weights.T)).cumprod()

    # method to normalize sum of weights to 1
    def normalize(self, _weights):
        return _weights / np.sum(_weights)

    # method to get a diagonal multiplication of two arrays
    # equivalent to np.diag(np.dot(A, B))
    def diag_mul(self, A, B):
        return (A * B.T).sum(-1)

    def conversion(self, _all_weights, _portfolio_return):
        self.all_weights = pd.DataFrame(_all_weights, index=self.final_time, columns=self.asset_name)
        self.portfolio_return = pd.DataFrame(_portfolio_return, index=self.final_time, columns=["Returns"])

    # calculate the variance based on the price
    def volatility(self):
        pass

    # Calculate Sharpe Ratio
    def sharpe_ratio(self):
        # self.portfolio_sharpe_ratio = ((self.portfolio_return - risk_free_rate) / (self.portfolio_risk ** 0.5))
        pass

    # return maximum drawdown
    def maximum_drawdown(self):
        return min(self.portfolio_return)

    # return summary of the portfolio
    def summary(self):
        pass

    # drop weights below a certain threshold
    def round_weights(self, _all_weights, threshold=1e-6):
        new_all_weights = np.where(_all_weights < threshold, 0, _all_weights)
        return np.apply_along_axis(lambda x: x / np.sum(x), 1, new_all_weights)

    # optimize the weight that maximizes the returns
    def optimize(self, _optimize_array, _solver=None):
        length_of_time = _optimize_array.shape[0]
        number_of_assets = _optimize_array.shape[1]
        if length_of_time == 1:
            best_idx = np.argmax(_optimize_array)
            weight = np.zeros(number_of_assets)
            weight[best_idx] = 1
            return weight

        # initialize weights
        weights = cp.Variable(self.number_of_assets)

        # used cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        portfolio_return = cp.sum(cp.log(_optimize_array * weights))

        # Optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [
            cp.sum(weights) == 1,
            weights <= 1,
            weights >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        if _solver:
            problem.solve(solver=_solver)
        else:
            problem.solve()
        return weights.value

        # optimize the weight that minimizes the l2 norm
    def simplex_projection(self, _optimize_weight):
        """

        :param _optimize_weight:
        :return:
        """
        # initialize weights
        weights = cp.Variable(self.number_of_assets)

        # used cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        l2_norm = cp.norm(weights - _optimize_weight)

        # Optimization objective and constraints
        allocation_objective = cp.Minimize(l2_norm)
        allocation_constraints = [
                cp.sum(weights) == 1,
                weights <= 1,
                weights >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
                objective=allocation_objective,
                constraints=allocation_constraints
        )
        problem.solve(solver=cp.SCS)
        return weights.value

    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    def generate_simplex(self, _number_of_portfolio, _number_of_assets):
        """

        :param _number_of_portfolio:
        :param _number_of_assets:
        :return:
        """
        simplex = np.sort(np.random.random((_number_of_portfolio, _number_of_assets - 1)))
        simplex = np.diff(np.hstack([np.zeros((_number_of_portfolio, 1)), simplex, np.ones((_number_of_portfolio, 1))]))
        return simplex.T


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
