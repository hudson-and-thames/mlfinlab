"""
Tests Universal Portfolio.
"""

from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.up import UP


class TestUniversalPortfolio(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the UP class.
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

    def test_up_solution(self):
        """
        Test the calculation of UP weights.
        """
        # Initialize UP.
        up1 = UP(2)
        # Allocates asset prices to UP.
        up1.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(up1.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_up_progress_solution(self):
        """
        Tests that UP prints progress bar.
        """
        # Initialize UP.
        up2 = UP(2)
        # Allocates asset prices to UP.
        up2.allocate(self.data, verbose=True)

    def test_up_uniform_solution(self):
        """
        Tests UP with uniform capital allocation.
        """
        # Initialize UP.
        up3 = UP(2, weighted='uniform')
        # Allocates asset prices to UP.
        up3.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(up3.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_up_top_k_solution(self):
        """
        Tests UP with top-k experts capital allocation.
        """
        # Initialize UP.
        up4 = UP(5, weighted='top-k', k=2)
        # Allocates asset prices to UP.
        up4.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(up4.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_up_wrong_method(self):
        """
        Tests ValueError if the method is not 'hist_performance', 'uniform', or 'top-k'.
        """
        # Initialize UP.
        up5 = UP(5, weighted='random', k=2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            up5.allocate(self.data)

    def test_up_recalculate_solution(self):
        """
        Tests recalculate method in UP.
        """
        # Initialize UP.
        up6 = UP(3, weighted='top-k', k=2)
        # Allocates asset prices to UP.
        up6.allocate(self.data)
        # Recalculate with k=1.
        up6.recalculate_k(1)
        # Create np.array of all_weights.
        all_weights = np.array(up6.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_up_recalculate_error(self):
        """
        Tests ValueError if k is greater number of experts for recalculate.
        """
        # Initialize UP.
        up7 = UP(3, weighted='top-k', k=2)
        # Allocates asset prices to UP.
        up7.allocate(self.data)
        with self.assertRaises(ValueError):
            # Recalculate will raise ValueError.
            up7.recalculate_k(4)

    def test_up_recalculate1_error(self):
        """
        Tests ValueError if k is not an integer for recalculate.
        """
        # Initialize UP.
        up8 = UP(3, weighted='top-k', k=2)
        # Allocates asset prices to UP.
        up8.allocate(self.data)
        with self.assertRaises(ValueError):
            # Recalculate will raise ValueError.
            up8.recalculate_k(1.5)

    def test_up_recalculate2_error(self):
        """
        Tests ValueError if k is not greater than or equal to 1.
        """
        # Initialize UP.
        up9 = UP(3, weighted='top-k', k=2)
        # Allocates asset prices to UP.
        up9.allocate(self.data)
        with self.assertRaises(ValueError):
            # Recalculate will raise ValueError.
            up9.recalculate_k(0)
