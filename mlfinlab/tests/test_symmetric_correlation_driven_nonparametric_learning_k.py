"""
Tests Symmetric Correlation Driven Nonparametric Learning - K.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.scornk \
    import SCORNK


class TestSymmetricCorrelationDrivenNonparametricLearningK(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Symmetric Correlation Driven Nonparametric Learning - K class.
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

    def test_scorn_k_solution(self):
        """
        Test the calculation of SCORN-K.
        """
        # Initialize SCORN-K.
        scorn_k = SCORNK(window=2, rho=2, k=1)
        # Allocates asset prices to SCORN-K.
        scorn_k.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(scorn_k.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_scorn_k_window_error(self):
        """
        Tests ValueError if window is not an integer or less than 1.
        """
        # Initialize SCORN-K.
        scorn_k1 = SCORNK(window=2.5, rho=2, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            scorn_k1.allocate(self.data)

        # Initialize SCORN-K.
        scorn_k2 = SCORNK(window=0, rho=2, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            scorn_k2.allocate(self.data)

    def test_scorn_k_rho_error(self):
        """
        Tests ValueError if rho is not an integer or less than 1.
        """
        # Initialize SCORN-K.
        scorn_k3 = SCORNK(window=2, rho=2.5, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            scorn_k3.allocate(self.data)

        # Initialize SCORN-K.
        scorn_k4 = SCORNK(window=2, rho=0, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            scorn_k4.allocate(self.data)

    def test_scorn_k_k_error(self):
        """
        Tests ValueError if k is greater than window * rho, greater than 1, or an integer.
        """
        # Initialize SCORN-K.
        scorn_k5 = SCORNK(window=2, rho=2, k=5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            scorn_k5.allocate(self.data)

        # Initialize SCORN-K.
        scorn_k6 = SCORNK(window=2, rho=2, k=1.5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            scorn_k6.allocate(self.data)

        # Initialize SCORN-K.
        scorn_k7 = SCORNK(window=2, rho=2, k=0)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            scorn_k7.allocate(self.data)
