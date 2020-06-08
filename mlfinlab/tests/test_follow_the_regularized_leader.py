"""
Tests Follow the Regularized Leader.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.ftrl import FTRL


class TestFollowTheRegularizedLeader(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Follow the Regularized Leader class.
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

    def test_ftrl_solution(self):
        """
        Test calculation of follow the regularized leader.
        """
        # Initialize FTRL.
        ftrl = FTRL(beta=0.2)
        # Allocate asset prices to FTRL.
        ftrl.allocate(self.data, resample_by='Y')
        all_weights = np.array(ftrl.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_beta_0_solution(self):
        """
        Test calculation of follow the regularized leader for beta value of 0.
        """
        # Initialize FTRL.
        ftrl = FTRL(beta=0)
        # Allocate asset prices to FTRL.
        ftrl.allocate(self.data, resample_by='Y')
        all_weights = np.array(ftrl.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_beta_1_solution(self):
        """
        Test calculation of follow the regularized leader for beta value of 1.
        """
        # Initialize FTRL.
        ftrl = FTRL(beta=1)
        # Allocate asset prices to FTRL.
        ftrl.allocate(self.data, resample_by='Y')
        all_weights = np.array(ftrl.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_beta_10_solution(self):
        """
        Test calculation of follow the regularized leader for beta value of 10.
        """
        # Initialize FTRL.
        ftrl = FTRL(beta=10)
        # Allocate asset prices to FTRL.
        ftrl.allocate(self.data, resample_by='Y')
        all_weights = np.array(ftrl.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
