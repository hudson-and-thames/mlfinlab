"""
Tests Online Portfolio Selection (OLPS).
"""

from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import OLPS


class TestOLPS(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Test different functions of the OLPS class.
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
        Test the calculation of OLPS weights and ensure that weights sum to 1.
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
        Test that the user inputted weights have matching dimensions as the data's.
        """
        # initialize OLPS
        olps1 = OLPS()
        # weight length of 1 does not match data.shape[1]
        with self.assertRaises(ValueError):
            olps1.allocate(self.data, weights=[1])

    def test_olps_weight_incorrect_sum(self):
        """
        Test ValueError if the user inputted weights do not sum to one.
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
        Test ValueError if the user inputted data is not a dataframe.
        """
        with self.assertRaises(ValueError):
            # initialize OLPS
            olps3 = OLPS()
            # wrong data format
            olps3.allocate(self.data.values)

    def test_olps_index_error(self):
        """
        Test ValueError if the passing dataframe is not indexed by date.
        """
        # initialize OLPS
        olps4 = OLPS()
        # index resetted
        data = self.data.reset_index()
        with self.assertRaises(ValueError):
            olps4.allocate(data)

    def test_user_weight(self):
        """
        Test that OLPS works if the user inputs their own weight.
        """
        # user weight
        weight = np.zeros(self.data.shape[1])
        weight[0] = 1
        # initialize OLPS
        olps5 = OLPS()
        # allocates self.data to OLPS
        olps5.allocate(self.data, weights=weight)
        # create np.array of all_weights
        all_weights = np.array(olps5.all_weights)
        # checks if all weights sum to 1
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_uniform_weight(self):
        """
        Test that uniform weights return equal allocation of weights.
        """
        olps6 = OLPS()
        olps6.allocate(self.data)
        olps6_uni_weight = olps6.uniform_weight()
        np.testing.assert_almost_equal(olps6_uni_weight, np.array(olps6.all_weights)[0])

    def test_normalize(self):
        """
        Test that weights sum to 1.
        """
        olps7 = OLPS()
        random_weight = np.ones(3)
        normalized_weight = olps7.normalize(random_weight)
        np.testing.assert_almost_equal(normalized_weight, random_weight / 3)