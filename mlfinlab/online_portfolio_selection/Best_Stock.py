from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class Best_Stock(OLPS):
    """

    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()


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

        # index of stock that incresed the most
        best_idx = np.argmax(cumulative_product[-1])
        self.weights = np.zeros(number_of_assets)
        self.weights[best_idx] = 1

        # initialize self.all_weights
        self.all_weights = self.weights
        self.portfolio_return = np.dot(self.weights, cumulative_product[0])

        # Run the Algorithm
        for t in range(1, time_period):
            self.run(self.weights)
            self.portfolio_return = np.vstack((self.portfolio_return, np.dot(self.weights, cumulative_product[t])))

    def run(self, _weights):
        # weights never change because you're just holding onto them, so this effectively becomes the same as OLPS run method
        super(Best_Stock, self).run(_weights)


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    best_stock = Best_Stock()
    best_stock.allocate(asset_names=names, asset_prices=stock_price)


if __name__ == "__main__":
    main()