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

        # Actual weight calculation

        # if weights is none, put 1 on a random stock
        if weights is None:
            self.weights = np.zeros(number_of_assets)
            self.weights[np.random.randint(0, number_of_assets - 1)] += 1
        else:
            self.weights = weights

        # initialize self.all_weights
        self.all_weights = self.weights
        self.portfolio_return = np.dot(self.weights, relative_price[0])

        # Run the Algorithm
        for t in range(1, time_period):
            self.run(self.weights)
            self.portfolio_return = np.vstack((self.portfolio_return, np.dot(self.weights, relative_price[t])))

        # convert everything to make presentable
        # convert to dataframe
        self.conversion(_all_weights=self.all_weights,_portfolio_return=self.portfolio_return,_index=idx,_asset_names=asset_names)

    # for this one, it doesn't matter but for subsequent complex selection problems, we might have to include a
    # separate run method for each iteration and not clog the allocate method.
    # after calculating the new weight add that to the all weights
    def run(self, weights):
        # update weights according to a certain algorithm
        new_weights = weights
        self.weights = new_weights
        self.all_weights = np.vstack((self.all_weights, self.weights))
        return self.weights

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
        self.portfolio_return = pd.DataFrame(_portfolio_return, index=_index,columns=["Relative Returns"])
    
    # calculate the returns based on portfolio weights
    def returns(self):
        pass

    # calculate the variance based on the price
    def volatility(self):
        pass

    # calculate the sharpe ratio based on the weights and returns
    def sharpe_ratio(self):
        pass

def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    initial_portfolio = OLPS()
    initial_portfolio.allocate(asset_names=names, asset_prices=stock_price)


if __name__ == "__main__":
    main()