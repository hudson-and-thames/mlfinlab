import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation


class TestMVO(unittest.TestCase):
    """
    Tests for the Mean Variance Optimisation module
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
        assert len(mvo.weights) > 0
        assert len(mvo.weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(mvo.weights), 1)
