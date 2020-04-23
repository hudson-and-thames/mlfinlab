"""
Tests Passive Aggressive Mean Reversion (PassiveAggressiveMeanReversion)
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import PassiveAggressiveMeanReversion


class TestPAMR(TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the PassiveAggressiveMeanReversion class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data = self.data.dropna(axis=1)

    def test_pamr_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with the original optimization method
        """

        pamr = PassiveAggressiveMeanReversion(optimization_method=0)
        pamr.allocate(self.data)
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

        pamr = PassiveAggressiveMeanReversion(optimization_method=1)
        pamr.allocate(self.data)
        all_weights = np.array(pamr.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_pamr2_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with the PassiveAggressiveMeanReversion-2 optimization method
        """

        pamr = PassiveAggressiveMeanReversion(optimization_method=2)
        pamr.allocate(self.data)
        all_weights = np.array(pamr.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)