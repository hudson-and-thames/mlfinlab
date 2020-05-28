"""
Tests Functional Correlation Driven Nonparametric Learning - K.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.fcornk \
    import FCORNK


class TestFunctionalCorrelationDrivenNonparametricLearningK(TestCase):
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Functional Correlation Driven Nonparametric Learning - K class.
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

    def test_fcorn_k_solution(self):
        """
        Test the calculation of FCORN-K.
        """
        # Initialize FCORN-K.
        fcorn_k = FCORNK(window=1, rho=1, lambd=1, k=1)
        # Allocates asset prices to FCORN-K.
        fcorn_k.allocate(self.data, resample_by='3M')
        # Create np.array of all_weights.
        all_weights = np.array(fcorn_k.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_fcorn_k_window_error(self):
        """
        Tests ValueError if window is not an integer or less than 1.
        """
        # Initialize FCORN-K.
        fcorn_k1 = FCORNK(window=2.5, rho=2, lambd=1, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k1.allocate(self.data)

        # Initialize FCORN-K.
        fcorn_k2 = FCORNK(window=0, rho=2, lambd=1, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k2.allocate(self.data)

    def test_fcorn_k_rho_error(self):
        """
        Tests ValueError if rho is not an integer or less than 1.
        """
        # Initialize FCORN-K.
        fcorn_k3 = FCORNK(window=2, rho=2.5, lambd=1, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k3.allocate(self.data)

        # Initialize FCORN-K.
        fcorn_k4 = FCORNK(window=2, rho=0, lambd=1, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k4.allocate(self.data)

    def test_fcorn_k_lambd_error(self):
        """
        Tests ValueError if lambd is not an integer or less than 1.
        """
        # Initialize FCORN-K.
        fcorn_k5 = FCORNK(window=2, rho=2, lambd=1.5, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k5.allocate(self.data)

        # Initialize FCORN-K.
        fcorn_k6 = FCORNK(window=2, rho=2, lambd=0, k=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k6.allocate(self.data)

    def test_fcorn_k_k_error(self):
        """
        Tests ValueError if k is not an integer of greater than window * rho * lambd
        """
        # Initialize FCORN-K.
        fcorn_k7 = FCORNK(window=2, rho=2, lambd=2, k=16)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k7.allocate(self.data)

        # Initialize FCORN-K.
        fcorn_k8 = FCORNK(window=2, rho=2, lambd=2, k=1.2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn_k8.allocate(self.data)
