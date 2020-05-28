"""
Tests Symmetric Correlation Driven Nonparametric Learning.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.scorn \
    import SCORN


class TestSymmetricCorrelationDrivenNonparametricLearning(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Symmetric Correlation Driven Nonparametric Learning class.
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

    def test_scorn_solution(self):
        """
        Test the calculation of SCORN.
        """
        # Initialize SCORN.
        scorn = SCORN(window=2, rho=0.5)
        # Allocates asset prices to SCORN.
        scorn.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(scorn.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
