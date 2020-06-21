"""
Test Eigenportfolio for Statistical Arbitrage module.
"""

import unittest
import os
import pandas as pd
import numpy as np

from mlfinlab.statistical_arbitrage import Eigenportfolio


class TestEigenportfolio(unittest.TestCase):
    """
    Test Eigenportfolio.
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

    def test_eigenportfolio(self):
        """
        Test eigenportfolio for all and rolling windows.
        """
        # Initialize StatArb.
        model = Eigenportfolio()
        model1 = Eigenportfolio()
        model2 = Eigenportfolio()
        model3 = Eigenportfolio()

        # Allocate data to model.
        model.allocate(self.data, pc_num=2, intercept=False)
        model1.allocate(self.data, pc_num=5, window=4, intercept=False)
        model2.allocate(self.data, pc_num=4, window=6)
        model3.allocate(self.data, pc_num=3)

    def test_value_error(self):
        """
        Tests ValueError for the user given inputs.
        """
        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if data is a pd.DataFrame.
            model.allocate(1, pc_num=5)

        # Create null pd.DataFrame.
        null_price = self.data.copy()
        null_price[:] = np.nan

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if data has null values.
            model.allocate(null_price, pc_num=5)

        # Create zero pd.DataFrame.
        zero_price = self.data.copy()
        zero_price[:] = 0

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if data has zero values.
            model.allocate(zero_price, pc_num=5)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if pc_num is an integer.
            model.allocate(self.data, pc_num=4.3)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if pc_num is positive.
            model.allocate(self.data, pc_num=-1)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if pc_num is less than the number of assets.
            model.allocate(self.data, pc_num=50)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if window is an integer.
            model.allocate(self.data, pc_num=3, window=0.6)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if window is non-negative
            model.allocate(self.data, pc_num=3, window=-1)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if window is less than the number of periods.
            model.allocate(self.data, pc_num=3, window=4000)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if window is less than the number of periods.
            model.allocate(self.data, pc_num=3, window=4000)

        with self.assertRaises(ValueError):
            # Initialize model.
            model = Eigenportfolio()
            # Check if intercept is a boolean.
            model.allocate(self.data, pc_num=3, window=20, intercept=1)
