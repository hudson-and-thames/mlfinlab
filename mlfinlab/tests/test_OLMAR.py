"""
Tests Online Moving Average Reversion (OLMAR)
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import OLMAR


class TestOLMAR(TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the OLMAR class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data = self.data.dropna(axis=1)

    def test_olmar_solution(self):
        """
        Test the calculation of online moving average reversion with the original reversion method
        """

        olmar = OLMAR(reversion_method=1)
        olmar.allocate(self.data)
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

        olmar = OLMAR(reversion_method=2)
        olmar.allocate(self.data)
        all_weights = np.array(olmar.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)