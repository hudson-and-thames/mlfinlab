# pylint: disable=protected-access
"""
Tests the Theory-Implied Correlation (TIC) algorithm.
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.tic import TIC


class TestNCO(unittest.TestCase):
    """
    Tests different functions of the TIC algorithm class.
    """

    def setUp(self):
        """
        Initialize and load data
        """

        project_path = os.path.dirname(__file__)

        # Loading the price series of ETFs
        price_data_path = project_path + '/test_data/stock_prices.csv'
        self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col="Date")

        # Loading the classification tree of ETFs
        classification_tree_path = project_path + '/test_data/classification_tree.csv'
        self.classification_tree = pd.read_csv(classification_tree_path)

    def test_get_linkage_corr(self):
        """
        Test the creation of a dendrogram.
        """

        tic = TIC()

        # Taking the first 5 ETFs for test purposes
        etf_prices = self.price_data.iloc[:, :5]
        etf_classification_tree = self.classification_tree.iloc[:5]

        # Expected dendrogram
        dend_expected = np.array([(1, 4, 0.10526126, 2), (5, 3, 0.23105119, 3),
                                  (0, 6, 0.40104189, 4), (7, 2, 0.59567056, 5)])

        # Calculationg simple correlation matrix for the TIC algorithm input
        etf_corr = etf_prices.corr()

        # Using the function
        dendrogram = tic.get_linkage_corr(etf_classification_tree, etf_corr)

        # Testing that the obtained dendrogram is right
        # Transforming to DataFrames to get same types
        np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dend_expected)),
                                       decimal=2)
