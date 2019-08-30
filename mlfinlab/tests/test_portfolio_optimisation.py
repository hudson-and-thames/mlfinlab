"""
Tests the different portfolio optimisation algorithms
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.cla import CLA
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation


class TestCLA(unittest.TestCase):
    """
    Tests different functions of the CLA class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="date")

    def test_cla_turning_points(self):
        """
        Test the calculation of CLA turning points
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data)
        for turning_point in cla.weights:
            assert turning_point
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)

    def test_cla_max_sharpe(self):
        """
        Test the calculation of maximum sharpe ratio weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='max_sharpe')
        assert cla.weights
        assert len(cla.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(cla.weights), 1)

    def test_cla_min_volatility(self):
        """
        Test the calculation for minimum volatility weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        assert cla.weights
        assert len(cla.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(cla.weights), 1)

    def test_cla_efficient_frontier(self):
        """
        Test the calculation of the efficient frontier solution
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='efficient_frontier')
        assert len(cla.means) == len(cla.sigma) and len(cla.sigma) == len(cla.weights)
        assert cla.sigma[-1] < cla.sigma[0] and cla.means[-1] < cla.means[0]  # higher risk = higher return

class TestHRP(unittest.TestCase):
    """
    Tests different functions of the HRP algorithm class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="date")

    def test_hrp(self):
        """
        Test the HRP algorithm
        """
        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        assert hrp.weights
        assert len(hrp.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(hrp.weights), 1)

    def test_quasi_diagnalization(self):
        """
        Test the quasi-diagnalisation step of HRP algorithm
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        assert hrp.ordered_indices == [12, 6, 14, 11, 5, 13, 3, 15, 7, 10, 17,
                                       18, 19, 4, 2, 0, 1, 16, 8, 9]

class TestMVO(unittest.TestCase):
    """
    Tests the different functions of the Mean Variance Optimisation class
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="date")

    def test_inverse_variance(self):
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data)
        assert mvo.weights
        assert len(mvo.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(mvo.weights), 1)
