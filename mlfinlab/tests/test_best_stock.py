"""
Tests Best Stock.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import BestStock


class TestBestStock(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Best Stock class.
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

    def test_best_stock_solution(self):
        """
        Tests that the weights sum to 1.
        """
        # Initialize BestStock.
        beststock = BestStock()
        # Allocates asset prices to BestStock.
        beststock.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(beststock.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_best_performing(self):
        """
        Tests that returning weights indicate the best performing asset.
        """
        # Initialize BestStock.
        beststock1 = BestStock()
        # Allocates asset prices to BestStock.
        beststock1.allocate(self.data)
        # Best performing asset calculated by dividing the last row by the first row.
        price_diff = np.array(self.data.iloc[-1] / self.data.iloc[0])
        # Index of best stock through manual calculation.
        idx_price_diff = np.argmax(price_diff)
        # Weight returned by BestStock.
        beststock_weight = np.array(beststock1.all_weights)[0]
        # Index of the best stock through BestStock.
        idx_best_stock = np.argmax(beststock_weight)
        # The indices should be identical.
        np.testing.assert_equal(idx_best_stock, idx_price_diff)

    def test_number_of_nonzero(self):
        """
        Tests that the weights returned have only one value that is non-zero.
        """
        # Initialize BestStock.
        beststock = BestStock()
        # Allocates asset prices to BestStock.
        beststock.allocate(self.data)
        # Weight returned by BestStock.
        beststock_weight = np.array(beststock.all_weights)[0]
        # There should be only one non-zero value for the weights.
        np.testing.assert_equal(np.count_nonzero(beststock_weight), 1)
