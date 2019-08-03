"""
Tests for the CLA module
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.cla import CLA


class TestCLA(unittest.TestCase):

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="date")

    def test_cla(self):
        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data)
        for turning_point in cla.weights:
            assert len(turning_point) > 0
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)

    def test_cla_max_sharpe(self):
        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='max_sharpe')
        assert len(cla.weights) > 0
        assert len(cla.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(cla.weights), 1)

    def test_cla_min_volatility(self):
        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        assert len(cla.weights) > 0
        assert len(cla.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(cla.weights), 1)

    def test_cla_efficient_frontier(self):
        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='efficient_frontier')
        assert len(cla.mu) == len(cla.sigma) and len(cla.sigma) == len(cla.weights)
        assert cla.sigma[-1] < cla.sigma[0] and cla.mu[-1] < cla.mu[0]  # higher risk = higher return
