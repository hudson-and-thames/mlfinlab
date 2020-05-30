"""
Tests Best Constant Rebalanced Portfolio.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import BCRP, CRP


class TestBestConstantRebalancedPortfolio(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    # pylint: disable=protected-access
    """
    Tests different functions of the Best Constant Rebalanced Portfolio class.
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

    def test_bcrp_solution(self):
        """
        Tests the calculation of best constant rebalanced portfolio weights.
        """
        # Initialize BCRP.
        bcrp = BCRP()
        # Allocates asset prices to BCRP.
        bcrp.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(bcrp.all_weights)
        # All weights for the strategy have to be the same.
        one_weight = all_weights[0]
        # iterate through all_weights to check that weights equal to the first weight.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_bcrp_returns(self):
        """
        Tests that BCRP returns are higher than other CRP's.
        """
        # Initialize BCRP
        bcrp1 = BCRP()
        # Allocates asset prices to BCRP.
        bcrp1.allocate(self.data, resample_by='M')
        # Get final returns for bcrp1.
        bcrp1_returns = np.array(bcrp1.portfolio_return)[-1]
        # Set an arbitray weight to test.
        weight = bcrp1._uniform_weight()
        # Initialize CRP.
        crp = CRP(weight)
        crp.allocate(self.data, resample_by='M')
        # Get final returns for CRP.
        crp_returns = np.array(crp.portfolio_return)[-1]
        # Check that CRP returns are lower than BCRP returns.
        np.testing.assert_array_less(crp_returns, bcrp1_returns)
