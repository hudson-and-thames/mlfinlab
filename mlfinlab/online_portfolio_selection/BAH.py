from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class BAH(OLPS):
    """
    This class implements the Buy and Hold strategy. It is reproduced with modification from the following paper:
    Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput. Surv. V, N, Article A (December YEAR),
    33 pages. DOI:http://dx.doi.org/10.1145/0000000.0000000.

    The Buy and Hold strategy one invests wealth among the market with an initial portfolio of weights and holds
    the portfolio till the end. The manager only buys the assets at the beginning of the first period and does
    not rebalance in subsequent periods.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

    # if there is user input, set it as that, if not we will return a uniform BAH
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

        # if user does not initiate a particular weight, give equal weights to every assets
        if weights is None:
            self.weights = np.ones(number_of_assets) / number_of_assets
        else:
            self.weights = weights

        # initialize self.all_weights
        self.all_weights = self.weights

        # sum of cumulative weights
        sum_cumulative_product = np.sum(cumulative_product, axis=1)

        # Run the Algorithm
        for t in range(1, time_period):
            self.run(cumulative_product[t], sum_cumulative_product[t])

        self.portfolio_return = np.dot(cumulative_product, self.all_weights[0])

    # update weights
    # although we're not rebalancing the portfolio, the weights themselves change because of the underlying price changes
    # we only need the cumulative product matrix to calculate the weights since we're just tracking the change
    # unnecessary run function but I kept it here so that it matches the other algorithms
    def run(self, _cumulative_product, _sum_cumulative_product):
        new_weights = _cumulative_product / _sum_cumulative_product
        self.weights = new_weights
        self.all_weights = np.vstack((self.all_weights, self.weights))
        return self.weights


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

        # self.weights = pd.DataFrame(self.weights)
        # self.weights.index = asset_names
        # self.weights = self.weights.T


def main():
    stock_price = pd.read_csv("../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    names = list(stock_price.columns)
    bah = BAH()
    bah.allocate(asset_names=names, asset_prices=stock_price)


if __name__ == "__main__":
    main()