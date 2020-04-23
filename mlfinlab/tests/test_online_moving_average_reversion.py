"""
Tests Online Moving Average Reversion (OnlineMovingAverageReversion)
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.mean_reversion.online_moving_average_reversion import OnlineMovingAverageReversion


class TestOnlineMovingAverageReversion(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=E1136
    """
    Tests different functions of the OnlineMovingAverageReversion class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """
        # sets project path to current directory
        project_path = os.path.dirname(__file__)
        # adds new data path to match stock_prices.csv data
        data_path = project_path + '/test_data/stock_prices.csv'
        # read_csv and parse dates
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        # dropna
        self.data = self.data.dropna(axis=1)

    def test_olmar_solution(self):
        """
        Test the calculation of online moving average reversion with the original reversion method
        """
        # initialize OLMAR
        olmar = OnlineMovingAverageReversion(reversion_method=1, epsilon=1, window=10)
        # sample by month
        olmar.allocate(self.data, resample_by='M')
        all_weights = np.array(olmar.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_olmar1_solution(self):
        """
        Test the calculation of online moving average reversion with the second reversion method
        """
        # initialize OLMAR
        olmar = OnlineMovingAverageReversion(reversion_method=2, epsilon=10, alpha=0.5)
        # sample by month
        olmar.allocate(self.data, resample_by='M')
        all_weights = np.array(olmar.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
