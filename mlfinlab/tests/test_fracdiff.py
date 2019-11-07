"""
Test various functions regarding chapter 5: Fractional Differentiation.
"""

import unittest
import os
import math
import numpy as np
import pandas as pd

from mlfinlab.features import fracdiff


class TestFractionalDifferentiation(unittest.TestCase):
    """
    Test get_weights, get_weights_ffd, frac_diff, and frac_diff_ffd
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

    def test_get_weights(self):
        """
        get_weights as implemented here matches the code in the book (Snippet 5.1).
        We test:
        1. if the first element of the weights vector is 1.0
        2. The length of the weights vector is same as the requested length
        """

        diff_amt = 0.9
        number_ele = 100
        weights = fracdiff.get_weights(diff_amt, size=number_ele)

        # Last value in the set is still the same
        self.assertTrue(weights[-1] == 1.0)

        # Size matches
        self.assertTrue(weights.shape[0] == number_ele)  # pylint: disable=unsubscriptable-object

    def test_get_weights_ffd(self):
        """
        get_weights_ffd as implemented here matches the code in the book (Snippet 5.2).
        We test:
        1. if the first element of the weights vector is 1.0
        2. The length of the weights vector is equal to 12

        """

        diff_amt = 0.9
        number_ele = 100
        thresh = 1e-3
        weights = fracdiff.get_weights_ffd(diff_amt, thresh=thresh, lim=number_ele)

        # Last value in the set is still the same
        self.assertTrue(weights[-1] == 1.0)

        # Size matches
        self.assertTrue(weights.shape[0] == 12)  # pylint: disable=unsubscriptable-object

    def test_frac_diff(self):
        """
        Assert that for any positive real number d,
        1. Length of the output is the same as the length of the input
        2. First element is NaN
        """
        data_series = self.data['close'].to_frame()

        for diff_amt in np.arange(0.1, 1, 0.1):
            fd_series = fracdiff.frac_diff(data_series, diff_amt=diff_amt)
            self.assertTrue(fd_series.shape[0] == len(data_series))
            self.assertTrue(isinstance(fd_series['close'][0], np.float64) and math.isnan(fd_series['close'][0]))

    def test_frac_diff_ffd(self):
        """
        Assert that for any positive real number d,
        1. Length of the output is the same as the length of the input
        2. First element is NaN
        """
        data_series = self.data['close'].to_frame()

        for diff_amt in np.arange(0.1, 1, 0.1):
            fd_series = fracdiff.frac_diff_ffd(data_series, diff_amt=diff_amt)
            self.assertTrue(fd_series.shape[0] == len(data_series))
            self.assertTrue(isinstance(fd_series['close'][0], np.float64) and math.isnan(fd_series['close'][0]))
