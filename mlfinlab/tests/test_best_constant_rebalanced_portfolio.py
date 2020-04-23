"""
Tests Best Constant Rebalanced Portfolio (BestConstantRebalancedPortfolio)
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import BestConstantRebalancedPortfolio


class TestBestConstantRebalancedPortfolio(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=E1136
    """
    Tests different functions of the BestConstantRebalancedPortfolio class.
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

    def test_bcrp_solution(self):
        """
        Test the calculation of best constant rebalanced portfolio weights.
        """
        # initialize BCRP
        bcrp = BestConstantRebalancedPortfolio()
        # allocates self.data to BCRP, resample by months for speed
        bcrp.allocate(self.data, resample_by='M')
        # create np.array of all_weights
        all_weights = np.array(bcrp.all_weights)
        # all weights have to be the same so make a default weight called one_weight
        one_weight = all_weights[0]
        # iterate through all to check weights equal original weight
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)
