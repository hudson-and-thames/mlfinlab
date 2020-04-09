from mlfinlab.online_portfolio_selection import CRP
from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class UP(CRP):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(UP, self).__init__()

    # if there is user input, set it as that, if not we will return a uniform CRP
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

        # Data Prep

        # same problems from OLPS

        # copy and pasted from OLPS beginning

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
        relative_price = self.relative_price_change(asset_prices)

        # cumulative product matrix
        cumulative_product = np.array(relative_price).cumprod(axis=0)
        """
        important part
        """
        # how many portfolios
        number_of_portfolios = 100

        # distribute wealth vector
        distribute_wealth = np.ones((number_of_portfolios, 1)) / number_of_portfolios

        # generate matrix of random weights
        random_weights = np.random.rand(number_of_assets, number_of_portfolios)

        # temporary check to compare to normal UCRP
        # random_weights = np.ones((number_of_portfolios, number_of_portfolios)) / number_of_portfolios

        # normalize each column by dividing by the sum to make the total sum of each column equal 1
        random_weights = np.apply_along_axis(lambda x: x / np.sum(x), 0, random_weights)

        # calculate the returns for all weights
        all_returns = np.dot(relative_price, random_weights)
        print(pd.DataFrame(all_returns))
        # calculate the portfolio returns
        portfolio_r = np.dot(all_returns, distribute_wealth)
        print(pd.DataFrame(portfolio_r))
        # calculate the final return value
        self.portfolio_return = portfolio_r.cumprod()
        print(self.portfolio_return)


        # for row in weight_options:
        #     self.portfolio_return += self.direct_returns(row, optimize_relative_price)
        # print(self.portfolio_return)
        # # if user does not initiate a particular weight, give equal weights to every assets
        # if weights is None:
        #     self.weights = np.ones(number_of_assets) / number_of_assets
        # else:
        #     self.weights = weights
        #
        # # initialize self.all_weights
        # self.all_weights = self.weights
        # self.portfolio_return = np.array([np.dot(self.weights, relative_price[0])])
        #
        # # Run the Algorithm
        # for t in range(1, time_period):
        #     self.run(self.weights, self.weights)
        #     self.returns(self.weights, relative_price[t], self.portfolio_return[self.portfolio_return.size - 1])
        #
        # self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
        #                 _asset_names=asset_names)

    # update weights
    # just copy and pasting the weights
    def run(self, _weights, _relative_price):
        self.weights = _weights
        self.all_weights = np.vstack((self.all_weights, self.weights))

    # directly calculate the final portfolio returns to optimize for computation
    def direct_returns(self, _weights, _relative_price):
        return np.prod(np.diagonal(np.dot(_relative_price, _weights)))


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    up = UP()
    up.allocate(asset_names=names, asset_prices=stock_price)
    print(up.all_weights)
    print(up.portfolio_return)


if __name__ == "__main__":
    main()
