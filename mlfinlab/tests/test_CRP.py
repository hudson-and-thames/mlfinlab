"""
Tests Constant Rebalanced Portfolio
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection import CRP


class TestCRP(TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the CRP class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")
        self.data = self.data.dropna(axis=1)

    def test_crp_solution(self):
        """
        Test the calculation of constant rebalanced portfolio weights.
        """

        crp = CRP()
        crp.allocate(self.data, resample_by='M')
        all_weights = np.array(crp.all_weights)
        one_weight = all_weights[0]
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            assert (weights == one_weight).all()
            np.testing.assert_almost_equal(np.sum(weights), 1)
        print(crp.all_weights)

