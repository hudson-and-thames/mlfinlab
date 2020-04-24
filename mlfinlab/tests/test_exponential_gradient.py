"""
Tests Exponential Gradient (ExponentialGradient).
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.momentum.exponential_gradient import ExponentialGradient


class TestExponentialGradient(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Exponential Gradient class.
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

    def test_mu_solution(self):
        """
        Test calculation of exponential gradient weights with multiplicative update rule.
        """
        # uses multiplicative update rule
        multiplicative_update = ExponentialGradient(eta=0.05, update_rule='MU')
        # resamples monthly
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
        # uses gradient projection update rule
        gradient_projection = ExponentialGradient(eta=0.1, update_rule='GP')
        # resamples monthly
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
        # uses expectation maximization update rule
        expectation_maximization = ExponentialGradient(eta=0.2, update_rule='EM')
        # resamples monthly
        expectation_maximization.allocate(self.data, resample_by='M')
        all_weights = np.array(expectation_maximization.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
