from mlfinlab.online_portfolio_selection import BCRP
from mlfinlab.online_portfolio_selection.olps_utils import *
import cvxpy as cp


class FTL(BCRP):
    """
    This class implements the Constant Rebalanced Portfolio strategy.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

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

        # find the best weights
        # optimize_array = relative_price - 1
        # optimize_array = optimize_array[1:]
        # self.optimize(optimize_array)

        # initialize self.weights
        self.initialize_weight(number_of_assets)

        # initialize self.all_weights
        self.all_weights = self.weights
        # next time will be equal weight
        self.all_weights = np.vstack((self.all_weights, self.weights))

        # Run the Algorithm starting from the third time because relative price is not just all 1's
        for t in range(2, time_period):
            self.run(relative_price[:t])

        self.portfolio_return = self.calculate_portfolio_returns(self.all_weights, relative_price)

        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
                        _asset_names=asset_names)

    # update weights
    # just copy and pasting the weights
    def run(self, _relative_price):
        self.optimize(_relative_price)
        self.all_weights = np.vstack((self.all_weights, self.weights))

def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    ftl = FTL()
    ftl.allocate(asset_names=names, asset_prices=stock_price)
    print(ftl.all_weights)
    print(ftl.portfolio_return)


if __name__ == "__main__":
    main()