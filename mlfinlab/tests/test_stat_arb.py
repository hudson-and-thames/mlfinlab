"""
Test StatArb.
"""

import unittest
import os
import pandas as pd
import numpy as np

from mlfinlab.statistical_arbitrage import StatArb


class TestStatArb(unittest.TestCase):
    """
    Test StatArb.
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

    def test_pairs_trading(self):
        """
        Test pairs trading for all and rolling windows.
        """
        # Initialize StatArb.
        model = StatArb()
        model1 = StatArb()
        model2 = StatArb()
        model3 = StatArb()

        # Allocate data to model.
        model.allocate(self.data.iloc[:, 0], self.data.iloc[:, 1], intercept=False)
        model1.allocate(self.data.iloc[:, 0], self.data.iloc[:, 1], window=4, intercept=False)
        model2.allocate(self.data.iloc[:, 0], self.data.iloc[:, 1], window=6)
        model3.allocate(self.data.iloc[:, 0], self.data.iloc[:, 1])

    def test_value_error(self):
        """
        Tests ValueError for the user given inputs.
        """
        price_x = self.data.iloc[:, 0]
        price_y = self.data.iloc[:, 1]
        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if price_x is a series.
            model.allocate(1, price_y)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if price_y is a series.
            model.allocate(price_x, 1)

        # Create null series.
        null_price = price_x.copy()
        null_price[:] = np.nan

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if price_x is null.
            model.allocate(null_price, price_y)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if price_y is null.
            model.allocate(price_x, null_price)

        # Create zero series.
        zero_price = price_x.copy()
        zero_price[:] = 0

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if price_x is null.
            model.allocate(zero_price, price_y)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if price_y is null.
            model.allocate(price_x, zero_price)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if price indices are the same.
            model.allocate(price_x[0:500], price_y[1:501])

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if window is negative.
            model.allocate(price_x, price_y, window=-1)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if window is an integer.
            model.allocate(price_x, price_y, window=0.5)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if window is less than the index.
            model.allocate(price_x, price_y, window=4000)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = StatArb()
            # Check if intercept is a boolean.
            model.allocate(price_x, price_y, intercept=1)

    def test_pinv_edge(self):
        """
        Tests pinv edge case for singular matrix.
        """
        # Initialize StatArb.
        model = StatArb()

        # Allocate data to model.
        model.allocate(self.data.iloc[:, 0], self.data.iloc[:, 0], window=1)
