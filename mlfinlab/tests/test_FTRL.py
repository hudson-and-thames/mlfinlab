"""
Tests Follow the Regularized Leader (FollowTheRegularizedLeader).
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import FollowTheRegularizedLeader


class TestFTRL(TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the FollowTheRegularizedLeader class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data = self.data.dropna(axis=1)

    def test_ftrl_solution(self):
        """
        Test the calculation of follow the regularized leader
        """

        ftrl = FollowTheRegularizedLeader(beta=0.1)
        ftrl.allocate(self.data)
        all_weights = np.array(ftrl.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

