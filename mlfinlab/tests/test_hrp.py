"""
Tests for the HRP module
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity


class TestCLA(unittest.TestCase):

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="date")

    def test_hrp(self):
        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        assert len(hrp.weights) > 0
        assert len(hrp.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(hrp.weights), 1)

    def test_quasi_diagnalization(self):
        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        assert hrp.ordered_indices == [12, 6, 14, 11, 5, 13, 3, 15, 7, 10, 17,
                                       18, 19, 4, 2, 0, 1, 16, 8, 9]
