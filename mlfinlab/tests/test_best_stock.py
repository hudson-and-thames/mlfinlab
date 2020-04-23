"""
Tests Best Stock (BestStock)
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
    Tests different functions of the BestStock class.
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

    def test_best_stock_solution(self):
        """
        Test the calculation of best stock weights and ensure that weights sum to 1.
        """
        # initialize BestStock
        beststock = BestStock()
        # allocates self.data to BestStock
        beststock.allocate(self.data)
        # create np.array of all_weights
        all_weights = np.array(beststock.all_weights)
        # checks if all weights sum to 1
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_best_performing(self):
        """
        Test that returning weights indicate the best performing asset.
        """
        # initialize BestStock
        beststock1 = BestStock()
        # allocates self.data to BestStock
        beststock1.allocate(self.data)
        # best performing asset calculated by dividing the last row by the first row
        price_diff = np.array(self.data.iloc[-1] / self.data.iloc[0])
        # index of best stock through manual calculation
        idx_price_diff = np.argmax(price_diff)
        # weight returned by beststock
        beststock_weight = np.array(beststock1.all_weights)[0]
        # index of best stock through BestStock
        idx_best_stock = np.argmax(beststock_weight)
        # compare the two weights
        np.testing.assert_equal(idx_best_stock, idx_price_diff)

    def test_number_of_nonzero(self):
        """
        Test that the weights returned have only one value that is non-zero.
        """
        # initialize BestStock
        beststock = BestStock()
        # allocates self.data to BestStock
        beststock.allocate(self.data)
        # weight returned by beststock
        beststock_weight = np.array(beststock.all_weights)[0]
        # compare the two weights
        np.testing.assert_equal(np.count_nonzero(beststock_weight), 1)
