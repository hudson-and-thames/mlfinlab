"""
Test functions from codependence module: correlation distances, mutual info, variation of information.
"""

import unittest
import numpy as np
import pandas as pd

from mlfinlab.codependence.correlation import squared_angular_distance, angular_distance, absolute_angular_distance, \
    distance_correlation


class TestCodependence(unittest.TestCase):
    """
    Test codependence module
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        state = np.random.RandomState(42)
        self.x = state.normal(size=1000)
        self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
        self.y_2 = abs(self.x) + state.normal(size=1000) / 5

    def test_correlations(self):
        """
        Test correlation based coefficients: angular (abs, square), distance correlation.
        """
        angular_dist = angular_distance(self.x, self.y_1)
        sq_angular_dist = squared_angular_distance(self.x, self.y_1)
        abs_angular_dist = absolute_angular_distance(self.x, self.y_1)
        dist_corr = distance_correlation(self.x, self.y_1)

        self.assertAlmostEqual(angular_dist, 0.67, delta=1e-2)
        self.assertAlmostEqual(abs_angular_dist, 0.6703, delta=1e-2)
        self.assertAlmostEqual(sq_angular_dist, 0.7, delta=1e-2)
        self.assertAlmostEqual(dist_corr, 0.529, delta=1e-2)

        dist_corr_y_2 = distance_correlation(self.x, self.y_2)
        self.assertAlmostEqual(dist_corr_y_2, 0.5216, delta=1e-2)
