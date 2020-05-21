"""
Tests Buy and Hold.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import BAH


class TestBuyAndHold(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=E1136
    """
    Tests different functions of the Buy and Hold class.
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

    def test_buy_and_hold_solution(self):
        """
        Tests the calculation of buy and hold weights.
        """
        # Initialize BAH.
        bah = BAH()
        # Allocates asset prices to BAH.
        bah.allocate(self.data)
        # Create np.array of all_weights.
        all_weights = np.array(bah.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_buy_two_assets(self):
        """
        Tests that weights are changing for a portfolio of two assets.
        """
        # Initialize weights with 0.5 on two random assets.
        weight = np.zeros(self.data.shape[1])
        weight[:2] = 0.5
        # Initialize BAH.
        bah1 = BAH()
        # Allocates asset prices to BAH.
        bah1.allocate(self.data, weight)
        # Get the second to last weight.
        last_bah1_weight = np.array(bah1.all_weights)[-1]
        # Relative price is calculated by dividing the second to last row by the first row.
        price_diff = np.array(self.data.iloc[-2] / self.data.iloc[0])
        # Calculate portfolio growth.
        new_last_weight = price_diff * weight
        # Normalize to sum the weight to one.
        norm_new_weight = new_last_weight/np.sum(new_last_weight)
        # Manual calculation should equal the weights returned by BAH.
        np.testing.assert_almost_equal(last_bah1_weight, norm_new_weight)

    def test_buy_five_assets(self):
        """
        Tests that weights are changing for a portfolio of five assets.
        """
        # Initialize weights with 0.2 on five random assets.
        weight = np.zeros(self.data.shape[1])
        weight[:2] = 0.5
        # Initialize BAH.
        bah2 = BAH()
        # Allocates asset prices to BAH.
        bah2.allocate(self.data, weight)
        # Get the second to last weight.
        last_bah2_weight = np.array(bah2.all_weights)[-1]
        # Relative price is calculated by dividing the second to last row by the first row.
        price_diff = np.array(self.data.iloc[-2] / self.data.iloc[0])
        # Calculate portfolio growth.
        new_last_weight = price_diff * weight
        # Normalize to sum the weight to one.
        norm_new_weight = new_last_weight/np.sum(new_last_weight)
        # Manual calculation should equal the weights returned by BAH.
        np.testing.assert_almost_equal(last_bah2_weight, norm_new_weight)
