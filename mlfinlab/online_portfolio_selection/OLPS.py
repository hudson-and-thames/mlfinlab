from mlfinlab.online_portfolio_selection.olps_utils import *


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
        # return asset time
        self.return_time = None
        self.return_number_of_time = None
        # relative return
        self.relative_return = None
        self.return_relative_return = None
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
        self.__check_asset(asset_prices, weights, portfolio_start, resample_by)

        # Data prep
        self.__initialize(asset_prices, weights, portfolio_start, resample_by)

        # Actual weight calculation
        # For future portfolios only change __run() to update the algorithms
        self.__run()

        # # Calculate Metrics
        # self.calculate_portfolio_returns(self.all_weights, self.relative_return)

        # convert everything to make presentable
        # convert to dataframe
        # self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
        #                 _asset_names=asset_names)

    # check for valid dataset
    # raise ValueError
    def __check_asset(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        # is the dataset actually valid
        pass
        # check _asset_prices is dataframe
        # check weights size is _asset_prices column size
        # _resample_by actually works
        # _portfolio_start is a valid number

    def __initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        # resample asset
        _asset_prices = _asset_prices.resample(_resample_by).last()

        # set portfolio start
        self.portfolio_start = _portfolio_start

        # set asset names
        self.asset_name = _asset_prices.columns

        # set time and returns time
        self.time = _asset_prices.index
        self.return_time = self.time[self.portfolio_start:]

        # calculate number of assets
        self.number_of_assets = self.asset_name.size

        # calculate number of time and number of time for returns
        self.number_of_time = self.time.size
        self.return_number_of_time = self.return_time.size

        # calculate relative returns
        self.relative_return = self.__relative_return(_asset_prices)

        if _weights is None:
            self.weights = self.__uniform_weight(self.number_of_assets)
        else:
            self.weights = _weights

        # set return_weights
        self.return_weights = np.zeros((self.return_time, self.number_of_assets))

        # set portfolio_return
        self.portfolio_return = np.zeros((self.return_time, self.number_of_assets))

    # calculate relative returns
    def __relative_return(self, _asset_prices):
        # percent change of each row
        # first row is blank because no change, so make it 0
        # add 1 to all values so that the values can be multiplied easily
        # change to numpy array
        relative_return = np.array(_asset_prices.pct_change().fillna(0) + 1)
        return relative_return

    # return uniform weights numpy array (1/n, 1/n, 1/n ...)
    def __uniform_weight(self, n):
        return np.ones(n) / n

    # for this one, it doesn't matter, but for subsequent complex selection problems, we might have to include a
    # separate run method for each iteration and not clog the allocate method.
    # after calculating the new weight add that to the all weights
    def __run(self, _weights, _relative_return):
        # Run the Algorithm
        for t in range(1, time_period):
            # update weights
            self.run(self.weights, relative_price[t - 1])


    def run(self, _past_weights, _past_relative_price):
        # no transactions, just moving weights around to reflect price difference
        new_weight = np.multiply(_past_weights, _past_relative_price)

        self.weights = self.normalize(new_weight)
        self.all_weights = np.vstack((self.all_weights, self.weights))

    # calculate portfolio returns
    def calculate_portfolio_returns(self, _all_weights, _relative_price):
        self.portfolio_return = np.diagonal(np.dot(_relative_price, _all_weights.T)).cumprod()

    # calculate the returns based on portfolio weights
    def returns(self, _current_weights, _current_relative_price, _previous_portfolio_return):
        new_returns = _previous_portfolio_return * np.dot(_current_weights, _current_relative_price)
        self.portfolio_return = np.vstack((self.portfolio_return, new_returns))

    # method to normalize sum of weights to 1
    def normalize(self, _weights):
        return _weights / np.sum(_weights)

    # method to get a diagonal multiplication of two arrays
    # equivalent to np.diag(np.dot(A, B))
    def diag_mul(self, A, B):
        return (A * B.T).sum(-1)

    def conversion(self, _all_weights, _portfolio_return, _index, _asset_names):
        self.all_weights = pd.DataFrame(_all_weights, index=_index, columns=_asset_names)
        self.portfolio_return = pd.DataFrame(_portfolio_return, index=_index, columns=["Relative Returns"])

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


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    initial_portfolio = OLPS()
    initial_portfolio.allocate(asset_names=names, asset_prices=stock_price)
    print(initial_portfolio.all_weights)
    print(initial_portfolio.portfolio_return)
    initial_portfolio.portfolio_return.plot()


if __name__ == "__main__":
    main()
