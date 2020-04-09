from mlfinlab.online_portfolio_selection.olps_utils import *
from mlfinlab.online_portfolio_selection.OLPS import OLPS


class CRP(OLPS):
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

        # index of stock that increased the most
        best_idx = np.argmax(cumulative_product[-1])
        self.weights = np.zeros(number_of_assets)
        self.weights[best_idx] = 1

        # initialize self.all_weights
        self.all_weights = self.weights
        self.portfolio_return = np.array([np.dot(self.weights, relative_price[0])])

        # Run the Algorithm
        for t in range(1, time_period):
            self.run(self.weights, relative_price[t-1])
            self.returns(self.weights, relative_price[t], self.portfolio_return[self.portfolio_return.size - 1])

        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
                        _asset_names=asset_names)

    # update weights
    # just copy and pasting the weights
    def run(self, _weights):
        # redundant code but necessary for latter parts of portfolio calculations
        new_weights = _weights
        # set self.weights as new_weights
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