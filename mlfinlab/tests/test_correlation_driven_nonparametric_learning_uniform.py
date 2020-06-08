"""
Tests Correlation Driven Nonparametric Learning Uniform.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.cornu import CORNU


class TestCorrelationDrivenNonparametricLearningUniform(TestCase):
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Correlation Driven Nonparametric Learning Uniform class.
    """

    def setUp(self):
        """
        Sets the file path for the tick data csv.
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices.csv'
        # Read csv, parse dates, and drop NaN.
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date").dropna(axis=1)

    def test_corn_u_solution(self):
        """
        Test the calculation of CORN-U.
        """
        # Initialize CORN-U.
        corn_u = CORNU(window=2, rho=0.5)
        # Allocates asset prices to CORN-U.
        corn_u.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(corn_u.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_corn_u_window_error(self):
        """
        Tests ValueError if window is not an integer or less than 1.
        """
        # Initialize CORN-U.
        corn_u1 = CORNU(window=2.5, rho=0.5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn_u1.allocate(self.data)

        # Initialize CORN-U.
        corn_u2 = CORNU(window=0, rho=0.5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn_u2.allocate(self.data)

    def test_corn_u_rho_error(self):
        """
        Tests ValueError if rho is less than -1 or more than 1.
        """
        # Initialize CORN-U.
        corn_u3 = CORNU(window=2, rho=-2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn_u3.allocate(self.data)

        # Initialize CORN-U.
        corn_u4 = CORNU(window=2, rho=2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn_u4.allocate(self.data)
