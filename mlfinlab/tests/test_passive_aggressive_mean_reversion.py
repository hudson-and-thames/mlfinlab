"""
Tests Passive Aggressive Mean Reversion (PassiveAggressiveMeanReversion)
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.mean_reversion.passive_aggressive_mean_reversion import PassiveAggressiveMeanReversion


class TestPassiveAggressiveMeanReversion(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=E1136
    """
    Tests different functions of the PassiveAggressiveMeanReversion class.
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

    def test_pamr_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with the original optimization method
        """

        pamr = PassiveAggressiveMeanReversion(epsilon=0.5, agg= 10, optimization_method=0)
        pamr.allocate(self.data, resample_by='M')
        all_weights = np.array(pamr.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_pamr1_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with the PassiveAggressiveMeanReversion-1 optimization method
        """

        pamr1 = PassiveAggressiveMeanReversion(epsilon=0.5, agg= 10, optimization_method=1)
        pamr1.allocate(self.data, resample_by='M')
        all_weights = np.array(pamr1.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_pamr2_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with the PassiveAggressiveMeanReversion-2 optimization method
        """

        pamr2 = PassiveAggressiveMeanReversion(epsilon=0.5, agg= 10, optimization_method=2)
        pamr2.allocate(self.data, resample_by='M')
        all_weights = np.array(pamr2.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
