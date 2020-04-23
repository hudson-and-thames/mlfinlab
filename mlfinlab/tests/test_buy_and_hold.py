"""
Tests Buy and Hold (BuyAndHold).
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import BuyAndHold


class TestBuyAndHold(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=E1136
    """
    Tests different functions of the BuyAndHold class.
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

    def test_buy_and_hold_solution(self):
        """
        Test the calculation of buy and hold weights.
        """

        # initialize BuyAndHold
        bah = BuyAndHold()
        # allocates self.data to BuyAndHold
        bah.allocate(self.data)
        # create np.array of all_weights
        all_weights = np.array(bah.all_weights)
        # checks if all weights sum to 1
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
