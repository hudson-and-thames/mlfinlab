from mlfinlab.online_portfolio_selection.olps_utils import *


# General OLPS class
class OLPS(object):
    # Initialize
    def __init__(self):
        """
        Constructor
        :param weights: (pd.DataFrame) final weight of portfolio
        :param all_weights: (pd.DataFrame) all weights of portfolio
        """
        self.weights = None
        self.all_weights = None
        self.portfolio_return = None

        # self.asset_prices = None
        # self.covariance_matrix = None
        # self.portfolio_risk = None
        # self.portfolio_sharpe_ratio = None
        # self.expected_returns = None
        # self.returns_estimator = ReturnsEstimation()

    # if a weight isn't specified, original OLPS will just buy a random stock and never rebalance afterwards
    def allocate(self,
                 asset_names,
                 asset_prices,
                 weights=None,
                 resample_by=None):
        """
        :param asset_names: (list) a list of strings containing the asset names
        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param weights: any weights
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """
        # Data prep

        # Some sort of initial check to make sure data fits the standards
        # asset name in right format
        # asset price in right format
        # resample_by in right format
        # weights add up to 1

        # calculate number of assets
        number_of_assets = len(asset_names)

        # split index and columns
        idx = asset_prices.index
        asset_names = asset_prices.columns

        # calculate number of time periods
        time_period = asset_prices.shape[0]

        # make asset_prices a numpy array (maybe faster for calculation)
        np_asset_prices = np.array(asset_prices)

        # calculate relative price i.e. week 1's price/week 0's price
        # relative_price is a dataframe
        relative_price = self.relative_price_change(asset_prices)

        # cumulative product matrix
        cumulative_product = np.array(relative_price).cumprod(axis=0)

        # Actual weight calculation

        # Buy one asset and never change
        self.weights = np.zeros(number_of_assets)
        self.weights[np.random.randint(0, number_of_assets - 1)] += 1

        # initialize self.all_weights
        self.all_weights = self.weights
        self.portfolio_return = np.array([np.dot(self.weights, relative_price[0])])

        # Run the Algorithm
        for t in range(1, time_period):
            # update weights
            self.run(self.weights, relative_price[t-1])
            # update portfolio_return
            self.returns(self.weights, relative_price[t], self.portfolio_return[self.portfolio_return.size - 1])

        # convert everything to make presentable
        # convert to dataframe
        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
                        _asset_names=asset_names)

    # for this one, it doesn't matter, but for subsequent complex selection problems, we might have to include a
    # separate run method for each iteration and not clog the allocate method.
    # after calculating the new weight add that to the all weights
    def run(self, _weights, _relative_price):
        # update weights according to a certain algorithm
        new_weights = _weights

        self.normalize_and_add(new_weights, _relative_price)

    # calculate the returns based on portfolio weights
    def returns(self, _weights, _relative_price, _portfolio_return):
        new_returns = _portfolio_return * (1 + np.dot(_weights, _relative_price - np.ones(len(_weights))))
        self.portfolio_return = np.vstack((self.portfolio_return, new_returns))

    def normalize_and_add(self, _weights, _relative_price):
        # normalization factor
        total_weights = np.dot(_weights, _relative_price)
        # calculate the change divided by the normalization factor
        _weights = np.multiply(_weights, _relative_price) / total_weights

        self.weights = _weights
        self.all_weights = np.vstack((self.all_weights, self.weights))

    def relative_price_change(self, asset_prices):
        # percent change of each row
        relative_price = asset_prices.pct_change()
        # first row is blank because no change, so make it 0
        relative_price = relative_price.fillna(0)
        # add 1 to all values so that the values can be multiplied easily
        relative_price = np.array(relative_price + 1)
        return relative_price

    def conversion(self, _all_weights, _portfolio_return, _index, _asset_names):
        self.all_weights = pd.DataFrame(_all_weights, index=_index, columns=_asset_names)
        self.portfolio_return = pd.DataFrame(_portfolio_return, index=_index, columns=["Relative Returns"])

    # calculate the variance based on the price
    def volatility(self):
        pass

    # calculate the sharpe ratio based on the weights and returns
    def sharpe_ratio(self):
        pass

    # Other idea that might be implemented later

    # Calculate covariance of returns or use the user specified covariance matrix
    # covariance_matrix = calculate_covariance(asset_names, asset_prices, covariance_matrix, resample_by, self.returns_estimator)

    # Calculate the expected returns if the user does not supply any returns
    # expected_asset_returns = calculate_expected_asset_returns(asset_prices, expected_asset_returns, resample_by)

    # Calculate the portfolio risk and return if it has not been calculated
    # self.portfolio_risk = calculate_portfolio_risk(self.portfolio_risk, covariance_matrix, self.weights)

    # Calculate the portfolio return
    # self.portfolio_return = calculate_portfolio_return(self.portfolio_return, self.weights, expected_asset_returns)

    # Calculate Sharpe Ratio
    # self.portfolio_sharpe_ratio = ((self.portfolio_return - risk_free_rate) / (self.portfolio_risk ** 0.5))

def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    initial_portfolio = OLPS()
    initial_portfolio.allocate(asset_names=names, asset_prices=stock_price)


if __name__ == "__main__":
    main()
