"""
Tests Universal Portfolio.
"""

from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import UniversalPortfolio


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
        # Initialize OLPS.
        up = UniversalPortfolio(2)
        # Allocates asset prices to OLPS.
        up.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(up.all_weights)
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
        # Initialize OLPS.
        up1 = UniversalPortfolio(2)
        # Allocates asset prices to OLPS.
        up1.allocate(self.data, verbose=True)

    def test_up_uniform_solution(self):
        """
        Tests UP with uniform capital allocation.
        """
        # Initialize OLPS.
        up = UniversalPortfolio(2, weighted='U')
        # Allocates asset prices to OLPS.
        up.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(up.all_weights)
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
        # Initialize OLPS.
        up = UniversalPortfolio(5, weighted='K', k=2)
        # Allocates asset prices to OLPS.
        up.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(up.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_up_wrong_method(self):
        """
        Tests ValueError if the method is not 'P', 'U', or 'K'.
        """
        # Initialize OLPS.
        up = UniversalPortfolio(5, weighted='JJ', k=2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            up.allocate(self.data)
