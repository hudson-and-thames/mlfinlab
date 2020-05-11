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

    def test_fcorn_window_error(self):
        """
        Tests ValueError if window is not an integer.
        """
        # Initialize FCORN.
        fcorn1 = FunctionalCorrelationDrivenNonparametricLearning(window=2.5, rho=0.5, lambd=1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn1.allocate(self.data)

    def test_fcorn_window1_error(self):
        """
        Tests ValueError if window is less than 1.
        """
        # Initialize FCORN.
        fcorn2 = FunctionalCorrelationDrivenNonparametricLearning(window=0, rho=0.5, lambd=2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn2.allocate(self.data)

    def test_fcorn_rho_error(self):
        """
        Tests ValueError if rho is less than -1.
        """
        # Initialize FCORN.
        fcorn3 = FunctionalCorrelationDrivenNonparametricLearning(window=2, rho=-2, lambd=4)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn3.allocate(self.data)

    def test_corn_rho1_error(self):
        """
        Tests ValueError if rho is less than -1.
        """
        # Initialize FCORN.
        fcorn4 = FunctionalCorrelationDrivenNonparametricLearning(window=2, rho=2, lambd=8)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            fcorn4.allocate(self.data)

    def test_sigmoid(self):
        """
        Tests Sigmoid Calculation.
        """
        # Initialize FCORN.
        fcorn5 = FunctionalCorrelationDrivenNonparametricLearning(window=2, rho=2, lambd=16)
        np.testing.assert_almost_equal(fcorn5.sigmoid(0), 0.5)
