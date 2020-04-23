"""
Tests Constant Rebalanced Portfolio (ConstantRebalancedPortfolio)
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import ConstantRebalancedPortfolio


class TestConstantRebalancedPortfolio(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=E1136
    """
    Tests different functions of the ConstantRebalancedPortfolio class.
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

    def test_default_crp_solution(self):
        """
        Test the calculation of constant rebalanced portfolio weights with default settings.
        """
        # initialize CRP
        crp = ConstantRebalancedPortfolio()
        # allocates self.data to CRP and resamble by month for speed
        crp.allocate(self.data, resample_by='M')
        # create np.array of all_weights
        all_weights = np.array(crp.all_weights)
        # all weights have to be the same so make a default weight called one_weight
        one_weight = all_weights[0]
        # iterate through all to check weights equal original weight
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_given_weights_crp_solution(self):
        """
        Test the calculation of constant rebalanced portfolio weights with weights given initially.
        """
        # create user input weights that place 1 on the first stock and 0 on the rest
        weights = np.zeros(self.data.shape[1])
        weights[0] = 1
        # initialize CRP
        crp = ConstantRebalancedPortfolio(weights)
        # allocates self.data to CRP and resamble by month for speed
        crp.allocate(self.data, resample_by='M')
        # create np.array of all_weights
        all_weights = np.array(crp.all_weights)
        # all weights have to be the same so make a default weight called one_weight
        one_weight = all_weights[0]
        # iterate through all to check weights equal original weight
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_given_allocate_weights_crp_solution(self):
        """
        Test the calculation of constant rebalanced portfolio weights with weights given in allocate.
        """
        # create user input weights that place 1 on the first stock and 0 on the rest
        weights = np.zeros(self.data.shape[1])
        weights[0] = 1
        # initialize CRP
        crp = ConstantRebalancedPortfolio()
        # allocates self.data to CRP and resamble by month for speed
        crp.allocate(self.data, weights, resample_by='M')
        # create np.array of all_weights
        all_weights = np.array(crp.all_weights)
        # all weights have to be the same so make a default weight called one_weight
        one_weight = all_weights[0]
        # iterate through all to check weights equal original weight
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)
