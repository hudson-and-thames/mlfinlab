"""
Tests Correlation Driven Nonparametric Learning.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.corn import CORN


class TestCorrelationDrivenNonparametricLearning(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Correlation Driven Nonparametric Learning class.
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

    def test_corn_solution(self):
        """
        Test the calculation of CORN.
        """
        # Initialize CORN.
        corn = CORN(window=2, rho=0.5)
        # Allocates asset prices to CORN.
        corn.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(corn.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_corn_window_error(self):
        """
        Tests ValueError if window is not an integer or less than 1.
        """
        # Initialize CORN.
        corn1 = CORN(window=2.5, rho=0.5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn1.allocate(self.data)

        # Initialize CORN.
        corn2 = CORN(window=0, rho=0.5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn2.allocate(self.data)

    def test_corn_rho_error(self):
        """
        Tests ValueError if rho is less than -1 or more than 1.
        """
        # Initialize CORN.
        corn3 = CORN(window=2, rho=-2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn3.allocate(self.data)

        # Initialize CORN.
        corn4 = CORN(window=2, rho=2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            corn4.allocate(self.data)
