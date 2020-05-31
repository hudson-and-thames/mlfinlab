"""
Tests Constant Rebalanced Portfolio.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import CRP


class TestConstantRebalancedPortfolio(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the CRP class.
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

    def test_default_crp_solution(self):
        """
        Tests the calculation of CRP weights with default settings.
        """
        # Initialize CRP.
        crp = CRP()
        # Allocates asset prices to CRP.
        crp.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(crp.all_weights)
        # All weights for the strategy have to be the same.
        one_weight = all_weights[0]
        # Iterate through all_weights to check that weights equal to the first weight.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_given_weights_crp_solution(self):
        """
        Tests the calculation of constant rebalanced portfolio weights with weights given initially.
        """
        # Create user input weights
        weights = np.zeros(self.data.shape[1])
        # Set 1 on the first stock and 0 on the rest.
        weights[0] = 1
        # Initialize CRP.
        crp = CRP(weights)
        # Allocates asset prices to CRP.
        crp.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(crp.all_weights)
        # All weights for the strategy have to be the same.
        one_weight = all_weights[0]
        # Iterate through all_weights to check that weights equal to the first weight.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_given_allocate_weights_crp_solution(self):
        """
        Test calculation of constant rebalanced portfolio weights with weights given in allocate.
        """
        # Create user input weights
        weights = np.zeros(self.data.shape[1])
        # Set 1 on the first stock and 0 on the rest.
        weights[0] = 1
        # Initialize CRP.
        crp = CRP()
        # Allocates asset prices to CRP.
        crp.allocate(self.data, weights, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(crp.all_weights)
        # All weights for the strategy have to be the same.
        one_weight = all_weights[0]
        # Iterate through all_weights to check that weights equal to the first weight.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)
