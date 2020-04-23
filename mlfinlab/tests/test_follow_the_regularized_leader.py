"""
Tests Follow the RegularizedLeader (FollowTheRegularizedLeader).
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.momentum.follow_the_regularized_leader import FollowTheRegularizedLeader


class TestFollowTheRegularizedLeader(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=E1136
    """
    Tests different functions of the Follow the Leader class.
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

    def test_ftrl_solution(self):
        """
        Test the calculation of follow the leader
        """
        # uses multiplicative update rule
        ftrl = FollowTheRegularizedLeader(beta=0.2)
        # resamples monthly
        ftrl.allocate(self.data, resample_by='M')
        all_weights = np.array(ftrl.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
