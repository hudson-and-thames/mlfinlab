from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class EG(OLPS):
    """

    """

    def __init__(self, eta=0.05, update_rule='EG'):
        """
        Constructor.
        """
        super().__init__()
        self.eta = eta
        self.update_rule = update_rule

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

        if weights is None:
            self.weights = np.ones(number_of_assets) / number_of_assets
        else:
            self.weights = weights

        # initialize self.all_weights
        self.all_weights = self.weights

        # Run the Algorithm
        for t in range(1, time_period):
            self.run(self.weights, relative_price[t - 1])

        self.portfolio_return = self.calculate_portfolio_returns(self.all_weights, relative_price)

        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
                        _asset_names=asset_names)

    def run(self, _past_weights, _past_relative_price):
        vector_mul = np.dot(_past_weights, _past_relative_price)
        if self.update_rule == 'EG':
            new_weight = _past_weights * np.exp(self.eta * _past_relative_price / vector_mul)
        elif self.update_rule == 'GP':
            new_weight = _past_weights + self.eta * (_past_relative_price - np.sum(_past_relative_price) / _past_relative_price.size) / vector_mul
        elif self.update_rule == 'EM':
            new_weight = _past_weights * (1 + self.eta * (_past_relative_price/vector_mul - 1))

        self.weights = self.normalize(new_weight)
        self.all_weights = np.vstack((self.all_weights, self.weights))


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    eg = EG(update_rule='EM')
    eg.allocate(asset_names=names, asset_prices=stock_price)
    print(eg.all_weights)
    print(eg.portfolio_return)


if __name__ == "__main__":
    main()
