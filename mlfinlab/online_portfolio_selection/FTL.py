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
        optimize_array = relative_price - 1
        optimize_array = optimize_array[1:]
        self.optimize(optimize_array)

        # initialize self.all_weights
        self.all_weights = self.weights
        self.portfolio_return = np.array([np.dot(self.weights, relative_price[0])])

        # Run the Algorithm
        for t in range(1, time_period):
            self.run(self.weights, self.weights)

        self.portfolio_return = self.calculate_portfolio_returns(self.all_weights, relative_price)

        self.conversion(_all_weights=self.all_weights, _portfolio_return=self.portfolio_return, _index=idx,
                        _asset_names=asset_names)

    # update weights
    # just copy and pasting the weights
    def run(self, _weights, _relative_price):
        super(BCRP, self).run(_weights, _relative_price)

    def optimize(self, _optimize_array):
        length_of_time = _optimize_array.shape[0]
        number_of_assets = _optimize_array.shape[1]
        # initialize weights
        weights = cp.Variable(number_of_assets)

        # used cp.log and cp.sum to make the cost function a convex function
        # multiplying continuous returns equates to summing over the log returns
        portfolio_return = cp.sum(cp.log(_optimize_array * weights + np.ones(length_of_time)))

        # Optimization objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [
                cp.sum(weights) == 1,
                weights <= 1,
                weights >= 0
        ]
        # Define and solve the problem
        problem = cp.Problem(
                objective=allocation_objective,
                constraints=allocation_constraints
        )
        problem.solve()
        self.weights = weights.value


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