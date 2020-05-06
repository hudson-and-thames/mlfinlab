"""
Tests Functional Correlation Driven Nonparametric Learning.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.pattern_matching.functional_correlation_driven_nonparametric_learning \
    import FunctionalCorrelationDrivenNonparametricLearning


class TestFunctionalCorrelationDrivenNonparametricLearning(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Functional Correlation Driven Nonparametric Learning class.
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

    def test_fcorn_solution(self):
        """
        Test the calculation of FCORN.
        """
        # Initialize FCORN.
        fcorn = FunctionalCorrelationDrivenNonparametricLearning(window=1, rho=0.5, lambd=10)
        # Allocates asset prices to FCORN.
        fcorn.allocate(self.data, resample_by='Y')
        # Create np.array of all_weights.
        all_weights = np.array(fcorn.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
