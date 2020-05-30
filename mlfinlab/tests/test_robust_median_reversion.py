"""
Tests Robust Median Reversion.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.rmr import RMR


class TestRobustMedianReversion(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Robust Median Reversion class.
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

    def test_rmr_solution(self):
        """
        Test the calculation of RMR with the original method.
        """
        # Initialize RMR.
        rmr = RMR(epsilon=1.1, n_iteration=10, window=3, tau=0.001)
        # Allocates asset prices to RMR.
        rmr.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(rmr.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_rmr_epsilon_error(self):
        """
        Tests ValueError if epsilon is greater than 1.
        """
        # Initialize RMR.
        rmr1 = RMR(epsilon=0.5, n_iteration=10, window=3, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr1.allocate(self.data)

    def test_rmr_n_iteration_error(self):
        """
        Tests ValueError if n_iteration is not an integer or less than 2.
        """
        # Initialize RMR.
        rmr2 = RMR(epsilon=1.2, n_iteration=1.5, window=3, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr2.allocate(self.data)

        # Initialize RMR.
        rmr3 = RMR(epsilon=1.2, n_iteration=1, window=3, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr3.allocate(self.data)

    def test_rmr_window_error(self):
        """
        Tests ValueError if window is not an integer or less than 2.
        """
        # Initialize RMR.
        rmr4 = RMR(epsilon=1.2, n_iteration=4, window=3.5, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr4.allocate(self.data)

        # Initialize RMR.
        rmr5 = RMR(epsilon=1.2, n_iteration=4, window=1, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr5.allocate(self.data)

    def test_rmr_break_solution(self):
        """
        Test the calculation of RMR with the break case in _calc_median.
        """
        # Initialize RMR.
        rmr6 = RMR(epsilon=1.1, n_iteration=10, window=3, tau=0.9)
        # Allocates asset prices to RMR.
        rmr6.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(rmr6.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_rmr_transform_non_mu(self):
        """
        Tests edge case for _transform non_mu edge case.
        """
        # pylint: disable=protected-access
        # pylint: disable=no-self-use
        # Initialize RMR.
        rmr7 = RMR(epsilon=1.1, n_iteration=10, window=3, tau=0.9)
        # Make an empty array.
        empty = np.zeros((1, 2))
        # Calculate edge case.
        rmr7._transform(empty, empty)

    def test_rmr_norm2_0_mu(self):
        """
        Tests edge case for norm2 = 0 in _transform method.
        """
        # Initialize RMR.
        rmr8 = RMR(epsilon=2, n_iteration=2, window=2, tau=0.9)
        # Make the data all ones.
        new_data = self.data
        new_data[:] = 1
        # Calculate edge case.
        rmr8.allocate(new_data, resample_by='M')

    def test_rmr_tau_error(self):
        """
        Tests ValueError if tau is less than 0 or greater than or equal to 1.
        """
        # Initialize RMR.
        rmr9 = RMR(epsilon=2, n_iteration=2, window=2, tau=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr9.allocate(self.data)

        # Initialize RMR.
        rmr10 = RMR(epsilon=2, n_iteration=2, window=2, tau=-1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr10.allocate(self.data)
