"""
Tests Exponential Gradient.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.eg import EG


class TestExponentialGradient(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Exponential Gradient class.
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

    def test_mu_solution(self):
        """
        Test calculation of exponential gradient weights with multiplicative update rule.
        """
        # Use multiplicative update rule.
        multiplicative_update = EG(update_rule='MU')
        # Allocates asset prices to MU.
        multiplicative_update.allocate(self.data, resample_by='M')
        all_weights = np.array(multiplicative_update.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_gp_solution(self):
        """
        Test calculation of exponential gradient weights with gradient projection update rule.
        """
        # Use gradient projection update rule.
        gradient_projection = EG(update_rule='GP')
        # Allocates asset prices to GP.
        gradient_projection.allocate(self.data, resample_by='M')
        all_weights = np.array(gradient_projection.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_em_solution(self):
        """
        Test calculation of exponential gradient weights with expectation maximization update rule.
        """
        # Use expectation maximization update rule.
        expectation_maximization = EG(update_rule='EM')
        # Allocates asset prices to EM.
        expectation_maximization.allocate(self.data, resample_by='M')
        all_weights = np.array(expectation_maximization.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_wrong_update(self):
        """
        Tests ValueError if the passing update rule is not correct.
        """
        # Put in incorrect update rule.
        expectation_maximization = EG(update_rule='SS')
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            expectation_maximization.allocate(self.data, resample_by='M')
