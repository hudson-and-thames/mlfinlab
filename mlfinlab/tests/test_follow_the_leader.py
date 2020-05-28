"""
Tests Follow the Leader.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.ftl import FTL


class TestFollowTheLeader(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    # pylint: disable=protected-access
    """
    Tests different functions of the Follow the Leader class.
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

    def test_ftl_solution(self):
        """
        Test the calculation of follow the leader.
        """
        # Initialize FTL.
        ftl = FTL()
        # Allocate asset prices to FTL.
        ftl.allocate(self.data, resample_by='M')
        all_weights = np.array(ftl.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_ftl_first_weight(self):
        """
        Tests that the weights calculated for the first time period is uniform.
        """
        # Initialize FTL.
        ftl1 = FTL()
        # Allocate asset prices to FTL.
        ftl1.allocate(self.data, resample_by='M')
        all_weights = np.array(ftl1.all_weights)
        # Get uniform weights.
        uniform_weight = ftl1._uniform_weight()
        # Compare the first weight to uniform weights.
        np.testing.assert_almost_equal(uniform_weight, all_weights[0])
