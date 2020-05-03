"""
Tests Robust Median Reversion.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.mean_reversion.robust_median_reversion import RobustMedianReversion


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
        rmr = RobustMedianReversion(epsilon=1.1, n_iteration=10, window=3, tau=0.001)
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
        rmr1 = RobustMedianReversion(epsilon=0.5, n_iteration=10, window=3, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr1.allocate(self.data)

    def test_rmr_n_iteration_error(self):
        """
        Tests ValueError if n_iteration is not an integer.
        """
        # Initialize RMR.
        rmr2 = RobustMedianReversion(epsilon=1.2, n_iteration=1.5, window=3, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr2.allocate(self.data)

    def test_rmr_n_iteration1_error(self):
        """
        Tests ValueError if n_iteration is less than 2.
        """
        # Initialize RMR.
        rmr3 = RobustMedianReversion(epsilon=1.2, n_iteration=1, window=3, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr3.allocate(self.data)

    def test_rmr_window_error(self):
        """
        Tests ValueError if window is not an integer.
        """
        # Initialize RMR.
        rmr4 = RobustMedianReversion(epsilon=1.2, n_iteration=4, window=3.5, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr4.allocate(self.data)

    def test_rmr_window1_error(self):
        """
        Tests ValueError if window is less than 2.
        """
        # Initialize RMR.
        rmr5 = RobustMedianReversion(epsilon=1.2, n_iteration=4, window=1, tau=0.001)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            rmr5.allocate(self.data)

    def test_rmr_break_solution(self):
        """
        Test the calculation of RMR with the break case in _calc_median.
        """
        # Initialize RMR.
        rmr6 = RobustMedianReversion(epsilon=1.1, n_iteration=10, window=3, tau=10000)
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
    #
    # def test_rmr_transform(self):
    #     """
    #     Test the eta edge case in _transform method for RMR.
    #     """
    #     temp_data = self.data
    #     temp_data[:] = 1
    #     # Initialize RMR.
    #     rmr7 = RobustMedianReversion(epsilon=1.1, n_iteration=10, window=3, tau=0.001)
    #     # Allocates asset prices to RMR.
    #     rmr7.allocate(temp_data, resample_by='M')
    #     # Create np.array of all_weights.
    #     all_weights = np.array(rmr7.all_weights)
    #     # Check if all weights sum to 1.
    #     for i in range(all_weights.shape[0]):
    #         weights = all_weights[i]
    #         assert (weights >= 0).all()
    #         assert len(weights) == self.data.shape[1]
    #         np.testing.assert_almost_equal(np.sum(weights), 1)
