"""
Tests Online Portfolio Selection (OLPS).
"""

from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import OLPS


class TestOLPS(TestCase):
    """
    Tests different functions of the OLPS class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """
        # sets project path to current directory
        project_path = os.path.dirname(__file__)
        # adds new data path to match stock_prices.csv data
        data_path = project_path + '/test_data/stock_prices.csv'
        # read_csv and parse dates
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        # dropna
        self.data = self.data.dropna(axis=1)

    def test_olps_solution(self):
        """
        Test the calculation of OLPS weights.
        """
        # initialize OLPS
        olps = OLPS()
        # allocates self.data to OLPS
        olps.allocate(self.data)
        # create np.array of all_weights
        all_weights = np.array(olps.all_weights)
        # checks if all weights sum to 1
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_olps_weight_mismatch(self):
        """
        Test if the user inputted weights have matching dimensions as the data
        """
        # initialize OLPS
        olps1 = OLPS()
        # weight length of 1 does not match data.shape[1]
        with self.assertRaises(ValueError):
            olps1.allocate(self.data, weights=[1])

    def test_olps_weight_incorrect_sum(self):
        """
        Test if the user inputted weights do not sum to one
        """
        with self.assertRaises(AssertionError):
            # initialize OLPS
            olps2 = OLPS()
            # weights that sum to 0.4+0.4=0.8
            weight = np.zeros(self.data.shape[1])
            weight[0], weight[1] = 0.4, 0.4
            olps2.allocate(self.data, weight)

    def test_olps_incorrect_data(self):
        """
        Test that user inputted data is a dataframe
        """
        with self.assertRaises(ValueError):
            # initialize OLPS
            olps3 = OLPS()
            # wrong data format
            olps3.allocate(self.data.values)

    def test_olps_index_error(self):
        """
        Test ValueError on passing dataframe not indexed by date.
        """
        # initialize OLPS
        olps4 = OLPS()
        # index resetted
        data = self.data.reset_index()
        with self.assertRaises(ValueError):
            olps4.allocate(data)
