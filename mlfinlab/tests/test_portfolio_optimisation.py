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
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_cla_turning_points(self):
        """
        Test the calculation of CLA turning points
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data)
        weights = cla.weights.values
        weights[weights <= 1e-15] = 0 # Convert very very small numbers to 0
        for turning_point in weights:
            assert (turning_point >= 0).all()
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)

    def test_cla_max_sharpe(self):
        """
        Test the calculation of maximum sharpe ratio weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='max_sharpe')
        weights = cla.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cla_min_volatility(self):
        """
        Test the calculation for minimum volatility weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        weights = cla.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cla_efficient_frontier(self):
        """
        Test the calculation of the efficient frontier solution
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='efficient_frontier')
        assert len(cla.efficient_frontier_means) == len(cla.efficient_frontier_sigma) and \
               len(cla.efficient_frontier_sigma) == len(cla.weights.values)
        assert cla.efficient_frontier_sigma[-1] <= cla.efficient_frontier_sigma[0] and \
               cla.efficient_frontier_means[-1] <= cla.efficient_frontier_means[0]  # higher risk = higher return

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
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_hrp(self):
        """
        Test the HRP algorithm
        """
        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_quasi_diagnalization(self):
        """
        Test the quasi-diagnalisation step of HRP algorithm
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17,
                                       12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]

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
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_inverse_variance(self):
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)