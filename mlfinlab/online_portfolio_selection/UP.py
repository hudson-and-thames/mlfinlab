from mlfinlab.online_portfolio_selection import CRP
from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class UP(CRP):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def __init__(self, _number_of_iterations):
        """
        Constructor.
        """
        super(UP, self).__init__()
        self.number_of_iterations = _number_of_iterations

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
        number_of_portfolios = self.number_of_iterations

        # distribute wealth vector
        distribute_wealth = np.ones((number_of_portfolios, 1)) / number_of_portfolios

        # generate matrix of random weights
        random_weights = np.random.rand(number_of_assets, number_of_portfolios)

        # normalize each column by dividing by the sum to make the total sum of each column equal 1
        random_weights = np.apply_along_axis(lambda x: x / np.sum(x), 0, random_weights)

        # calculate the returns for all weights
        all_returns = np.dot(relative_price, random_weights)

        """
        algorithm is incorrect (changing things right now)
        """

        # # calculate cumulative portfolio return
        # all_returns = np.apply_along_axis(lambda x: x.cumprod(), 0, all_returns)
        # print(pd.DataFrame(all_returns))
        # # calculate changing portfolio allocation based on performance
        # portfolio_allocation = np.apply_along_axis(lambda x: x / np.sum(x), 1,
        #                                            np.vstack((np.ones(number_of_portfolios), all_returns))[:-1]).T
        # print(pd.DataFrame(portfolio_allocation))
        # # calculate portfolio change each week by getting the diagonal of the money allocation
        # portfolio_change = np.diag(np.dot(all_returns, portfolio_allocation))
        #
        # # cumulative returns
        # print(portfolio_change.cumprod())
        # # calculate all_weights for return purposes
        # self.all_weights = np.apply_along_axis(lambda x: x / np.sum(x), 0, np.dot(random_weights, all_returns.T))
        # # add first averaged weight to the beginning and pop the last
        # self.all_weights = np.hstack((self.all_weights[:, [0]], self.all_weights)).T[:-1]
        #
        # # calculate portfolio return by time
        # self.portfolio_return = 0
        #
        # self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
        #                 _asset_names=asset_names)


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    up = UP(10)
    up.allocate(asset_names=names, asset_prices=stock_price)
    print(up.all_weights)
    print(up.portfolio_return)


if __name__ == "__main__":
    main()
