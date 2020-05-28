"""
Tests Online Moving Average Reversion.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.olmar import OLMAR


class TestOnlineMovingAverageReversion(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Online Moving Average Reversion class.
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

    def test_olmar_solution(self):
        """
        Test the calculation of online moving average reversion with the original reversion method.
        """
        # Initialize OLMAR.
        olmar = OLMAR(reversion_method=1, epsilon=1, window=10)
        # Allocates asset prices to OLMAR.
        olmar.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(olmar.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_olmar1_solution(self):
        """
        Test the calculation of online moving average reversion with the second reversion method.
        """
        # Initialize OLMAR.
        olmar1 = OLMAR(reversion_method=2, epsilon=10, alpha=0.5)
        # Allocates asset prices to OLMAR.
        olmar1.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(olmar1.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_olmar_epsilon_error(self):
        """
        Tests ValueError if epsilon is below than 1.
        """
        # Initialize OLMAR.
        olmar2 = OLMAR(reversion_method=1, epsilon=0, window=10)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            olmar2.allocate(self.data)

    def test_olmar_window_error(self):
        """
        Tests ValueError if reversion method is 1 and window is less than 1.
        """
        # Initialize OLMAR.
        olmar3 = OLMAR(reversion_method=1, epsilon=2, window=0)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            olmar3.allocate(self.data)

    def test_olmar_alpha_error(self):
        """
        Tests ValueError if reversion method is 2 and alpha is greater than 1.
        """
        # Initialize OLMAR.
        olmar4 = OLMAR(reversion_method=2, epsilon=2, alpha=2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            olmar4.allocate(self.data)

    def test_olmar_alpha1_error(self):
        """
        Tests ValueError if reversion method is 2 and alpha is less than 1.
        """
        # Initialize OLMAR.
        olmar5 = OLMAR(reversion_method=2, epsilon=2, alpha=-1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            olmar5.allocate(self.data)

    def test_olmar_method_error(self):
        """
        Tests ValueError if reversion method is 2 and alpha is less than 1.
        """
        # Initialize OLMAR.
        olmar6 = OLMAR(reversion_method=4, epsilon=2, alpha=-1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            olmar6.allocate(self.data)

    def test_olmar_edge_case_error(self):
        """
        Tests that lambd returns 0 if predicted change is mean change.
        """
        # Initialize OLMAR.
        olmar7 = OLMAR(reversion_method=1, epsilon=2, window=1)
        no_change_data = self.data
        no_change_data.iloc[:] = 1
        olmar7.allocate(no_change_data, resample_by='M')
