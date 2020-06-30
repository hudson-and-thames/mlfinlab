"""
Tests Confidence Weighted Mean Reversion.
"""
from unittest import TestCase
import os
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.cwmr import CWMR


class TestConfidenceWeightedMeanReversion(TestCase):
    # pylint: disable=too-many-public-methods
    # pylint: disable=unsubscriptable-object
    """
    Tests different functions of the Confidence Weighted Mean Reversion class.
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

    def test_cwmr_solution(self):
        """
        Test the calculation of CWMR with the original method.
        """
        # Initialize CWMR.
        cwmr = CWMR(confidence=0.5, epsilon=0.5, method='var')
        # Allocates asset prices to CWMR.
        cwmr.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(cwmr.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cwmr_sd_solution(self):
        """
        Test the calculation of CWMR with the second method.
        """
        # Initialize CWMR.
        cwmr = CWMR(confidence=0.5, epsilon=0.5, method='sd')
        # Allocates asset prices to OLMAR.
        cwmr.allocate(self.data, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(cwmr.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cwmr_epsilon_error(self):
        """
        Tests ValueError if epsilon is greater than 1 or less than 0.
        """
        # Initialize CWMR.
        cwmr1 = CWMR(confidence=0.5, epsilon=2)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            cwmr1.allocate(self.data)

        # Initialize CWMR.
        cwmr2 = CWMR(confidence=0.5, epsilon=-1)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            cwmr2.allocate(self.data)

    def test_cwmr_confidence_error(self):
        """
        Tests ValueError if confidence is greater than 1 or less than 0.
        """
        # Initialize CWMR.
        cwmr3 = CWMR(confidence=2, epsilon=0.5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            cwmr3.allocate(self.data)

        # Initialize CWMR.
        cwmr4 = CWMR(confidence=-1, epsilon=0.5)
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            cwmr4.allocate(self.data)

    def test_cwmr_method_error(self):
        """
        Tests ValueError if method is not 'sd' or 'var'.
        """
        # Initialize CWMR.
        cwmr5 = CWMR(confidence=0.5, epsilon=0.5, method='normal')
        with self.assertRaises(ValueError):
            # Running allocate will raise ValueError.
            cwmr5.allocate(self.data)

    def test_cwmr_weights_solution(self):
        """
        Test the calculation of CWMR with given weights.
        """
        # Set weights.
        weight = np.zeros(self.data.iloc[0].shape)
        weight[0] = 1
        # Initialize CWMR.
        cwmr6 = CWMR(confidence=0.5, epsilon=0.5, method='var')
        # Allocates asset prices to CWMR.
        cwmr6.allocate(self.data, weights=weight, resample_by='M')
        # Create np.array of all_weights.
        all_weights = np.array(cwmr6.all_weights)
        # Check if all weights sum to 1.
        for i in range(all_weights.shape[0]):
            weights = all_weights[i]
            assert (weights >= 0).all()
            assert len(weights) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(weights), 1)
