"""
Tests Passive Aggressive Mean Reversion.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.pamr import \
    PAMR


class TestPassiveAggressiveMeanReversion(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Passive Aggressive Mean Reversion class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """
        # Set project path to current directory.
        project_path = os.path.dirname(__file__)
        # Add new data path to match stock_prices.csv data.
        data_path = project_path + '/test_data/stock_prices.csv'
        # Read csv, parse dates, and drop NaN.
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date").dropna(axis=1)

    def test_pamr_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with the original optimization
        method.
        """
        # Initialize PAMR.
        pamr = PAMR(optimization_method=0, epsilon=0.5, agg=10)
        # Allocates asset prices to PAMR.
        pamr.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(pamr.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_pamr1_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with PAMR-1 optimization method.
        """
        # Initialize PAMR-1.
        pamr1 = PAMR(optimization_method=1, epsilon=0.5, agg=10)
        # Allocates asset prices to PAMR.
        pamr1.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(pamr1.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_pamr2_solution(self):
        """
        Test the calculation of passive aggressive mean reversion with the PAMR-2 optimization method
        """
        # Initialize PAMR-2.
        pamr2 = PAMR(optimization_method=2, epsilon=0.5, agg=10)
        # Allocates asset prices to PAMR.
        pamr2.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(pamr2.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_pamr_epsilon_error(self):
        """
        Tests ValueError if epsilon is less than 0.
        """
        # Initialize PAMR.
        pamr3 = PAMR(optimization_method=2, epsilon=-1, agg=10)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            pamr3.allocate(self.data)

    def test_pamr_agg_error(self):
        """
        Tests ValueError if aggressiveness is less than 0.
        """
        # Initialize PAMR.
        pamr4 = PAMR(optimization_method=2, epsilon=0.5, agg=-5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            pamr4.allocate(self.data)

    def test_pamr_method_error(self):
        """
        Tests ValueError if optimization method is not 0, 1, or 2.
        """
        # Initialize PAMR.
        pamr5 = PAMR(optimization_method=5, epsilon=0.5, agg=10)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            pamr5.allocate(self.data)
