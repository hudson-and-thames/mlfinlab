"""
Tests Exponential Gradient (EG).
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import EG


class TestEG(TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the BAH class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data = self.data.dropna(axis=1)

    def test_mu_solution(self):
        """
        Test the calculation of exponential gradient weights with multiplicative update rule.
        """

        multiplicative_update = EG(update_rule='MU')
        multiplicative_update.allocate(self.data)
        all_weights = np.array(multiplicative_update.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_gp_solution(self):
        """
        Test the calculation of exponential gradient weights with gradient projection update rule.
        """

        gradient_projection = EG(update_rule='GP')
        gradient_projection.allocate(self.data)
        all_weights = np.array(gradient_projection.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_gp_solution(self):
        """
        Test the calculation of exponential gradient weights with expectation maximization update rule.
        """

        expectation_maximization = EG(update_rule='EM')
        expectation_maximization.allocate(self.data)
        all_weights = np.array(expectation_maximization.all_weights)
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
