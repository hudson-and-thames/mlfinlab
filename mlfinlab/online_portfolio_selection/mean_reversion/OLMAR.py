# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.online_portfolio_selection import OLPS


class OLMAR(OLPS):
    """
    This class implements the Online Moving Average Reversion Strategy.
    """

    def __init__(self, reversion_method, epsilon, window=None, alpha=None):
        """
        Constructor.
        """
        # check that epsilon is > 1
        # check that window is >= 1
        # check that alpha is (0,1)
        # check that reversion_method is either 1 or 2
        # if optimization_method == 2 then reversion_method doesn't matter
        self.epsilon = epsilon
        self.window = window
        self.alpha = alpha
        self.reversion_method = reversion_method
        self.moving_average_reversion = None
        super().__init__()

    # intialize moving average reversion
    def initialize(self, _asset_prices, _weights, _portfolio_start, _resample_by):
        """

        :param _asset_prices:
        :param _weights:
        :param _portfolio_start:
        :param _resample_by:
        :return:
        """
        super(OLMAR, self).initialize(_asset_prices, _weights, _portfolio_start, _resample_by)
        self.moving_average_reversion = self.calculate_rolling_moving_average(self.asset_prices, self.window,
                                                                              self.reversion_method, self.alpha)

    def update_weight(self, _weights, _relative_return, _time):
        """

        :param _weights:
        :param _relative_return:
        :param _time:
        :return:
        """
        if self.reversion_method == 1 and _time < self.window:
            return self.weights
        # calculate price relative
        predicted_change = self.moving_average_reversion[_time]
        # different OLMAR methods
        mean_relative = np.mean(predicted_change)
        mean_change = np.ones(self.number_of_assets) * mean_relative
        try:
            loss_fn = max(0, (self.epsilon - np.dot(_weights, predicted_change)))
        except:
            print(self.epsilon)
            print(_weights)
            print(predicted_change)
            raise ValueError()

        if loss_fn == 0:
            lambd = 0
        else:
            lambd = loss_fn / (np.linalg.norm(predicted_change - mean_change) ** 2)

        new_weights = _weights + lambd * (predicted_change - mean_change)
        if np.isnan(new_weights).any():
            raise ValueError()
        # if not in simplex domain
        return self.simplex_projection(new_weights)

    def simplex_projection(self,
                           _optimize_weight):
        """
        Calculates the simplex projection of the weights
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

        :param _optimize_weight: (np.array) a weight that will be projected onto the simplex domain
        :return weights.value: (np.array) simplex projection of the original weight
        """

        # return itself if already a simplex projection
        if np.sum(_optimize_weight) == 1 and np.all(_optimize_weight >= 0):
            return _optimize_weight

        # sort descending
        _mu = np.sort(_optimize_weight)[::-1]

        # adjusted sum
        adjusted_sum = np.cumsum(_mu) - 1

        # number
        j = np.arange(len(_optimize_weight)) + 1

        # condition
        cond = _mu - adjusted_sum / j > 0

        # define max rho
        rho = float(j[cond][-1])

        # define theta
        theta = adjusted_sum[cond][-1] / rho

        # calculate new weight
        new_weight = np.maximum(_optimize_weight - theta, 0)
        return new_weight

    def calculate_rolling_moving_average(self, _asset_prices, _window, _reversion_method, _alpha):
        """
        Calculates the rolling moving average for Online Moving Average Reversion

        :param _asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param _window: (int) number of market windows
        :param _reversion_method: (int) number that represents the reversion method
        :param _alpha: (int) exponential weight for the second reversion method
        :return rolling_ma: (np.array) rolling moving average for the given reversion method
        """
        # MAR-1 reversion method: Simple Moving Average
        if _reversion_method == 1:
            rolling_ma = np.array(_asset_prices.rolling(_window).apply(lambda x: np.sum(x) / x[0] / _window))
        # Mar-2 reversion method: Exponential Moving Average
        elif _reversion_method == 2:
            rolling_ma = np.array(_asset_prices.ewm(alpha=_alpha, adjust=False).mean() / _asset_prices)
        return rolling_ma
