"""
Test structural break tests: Chow-type, CUSUM, SADF
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.structural_breaks import (get_chow_type_stat, get_sadf, get_chu_stinchcombe_white_statistics)


# pylint: disable=unsubscriptable-object
from mlfinlab.structural_breaks.cusum import _get_values_diff
from mlfinlab.structural_breaks.sadf import get_betas


class TesStructuralBreaks(unittest.TestCase):
    """
    Test Chow-type, CUSUM, SADF tests
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/dollar_bar_sample.csv'
        self.data = pd.read_csv(self.path, index_col='date_time', parse_dates=[0])

    def test_chow_test(self):
        """
        Test get_chow_type_stat function
        """
        min_length = 10
        log_prices = np.log(self.data.close)
        stats = get_chow_type_stat(log_prices, min_length=min_length)

        # We drop first and last # of min_length values
        self.assertEqual(log_prices.shape[0] - min_length * 2, stats.shape[0])
        self.assertAlmostEqual(stats.max(), 0.179, delta=1e-3)
        self.assertAlmostEqual(stats.mean(), -0.653, delta=1e-3)
        self.assertAlmostEqual(stats[3], -0.6649, delta=1e-3)

    def test_chu_stinchcombe_value_diff_function(self):
        """
        Test the values diff hidden function.
        """
        # Test values diff function
        one_sided_diff = _get_values_diff(test_type='one_sided', series=pd.Series([1, 2, 3, 4, 5]), index=0, ind=1)
        two_sided_diff = _get_values_diff(test_type='two_sided', series=pd.Series([1, 2, 3, 4, 5]), index=0, ind=1)
        self.assertEqual(-1, one_sided_diff)
        self.assertEqual(1, two_sided_diff)
        self.assertRaises(ValueError, _get_values_diff, test_type='rubbish',
                          series=pd.Series([1, 2, 3, 4, 5]), index=0, ind=1)

    def test_chu_stinchcombe_white_test(self):
        """
        Test get_chu_stinchcombe_white_statistics function
        """

        log_prices = np.log(self.data.close)
        one_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='one_sided')
        two_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='two_sided')

        # For the first two values we don't have enough info
        self.assertEqual(log_prices.shape[0] - 2, one_sided_test.shape[0])
        self.assertEqual(log_prices.shape[0] - 2, two_sided_test.shape[0])

        self.assertAlmostEqual(one_sided_test.critical_value.max(), 3.265, delta=1e-3)
        self.assertAlmostEqual(one_sided_test.critical_value.mean(), 2.7809, delta=1e-3)
        self.assertAlmostEqual(one_sided_test.critical_value[20], 2.4466, delta=1e-3)

        self.assertAlmostEqual(one_sided_test.stat.max(), 3729.001, delta=1e-3)
        self.assertAlmostEqual(one_sided_test.stat.mean(), 836.509, delta=1e-3)
        self.assertAlmostEqual(one_sided_test.stat[20], 380.137, delta=1e-3)

        self.assertAlmostEqual(two_sided_test.critical_value.max(), 3.235, delta=1e-3)
        self.assertAlmostEqual(two_sided_test.critical_value.mean(), 2.769, delta=1e-3)
        self.assertAlmostEqual(two_sided_test.critical_value[20], 2.715, delta=1e-3)

        self.assertAlmostEqual(two_sided_test.stat.max(), 5518.519, delta=1e-3)
        self.assertAlmostEqual(two_sided_test.stat.mean(), 1264.582, delta=1e-3)
        self.assertAlmostEqual(two_sided_test.stat[20], 921.2979, delta=1e-3)

        self.assertRaises(ValueError, get_chu_stinchcombe_white_statistics, log_prices, 'rubbish text')

    def test_sadf_test(self):
        """
        Test get_sadf function
        """

        log_prices = np.log(self.data.close)
        lags_int = 5
        lags_array = [1, 2, 5, 7]
        min_length = 20

        linear_sadf = get_sadf(log_prices, model='linear', add_const=True, min_length=min_length, lags=lags_int)
        linear_sadf_no_const_lags_arr = get_sadf(log_prices, model='linear', add_const=False, min_length=min_length,
                                                 lags=lags_array)
        quadratic_sadf = get_sadf(log_prices, model='quadratic', add_const=True, min_length=min_length, lags=lags_int)

        sm_poly_1_sadf = get_sadf(log_prices, model='sm_poly_1', add_const=True, min_length=min_length, lags=lags_int)
        sm_poly_2_sadf = get_sadf(log_prices, model='sm_poly_2', add_const=True, min_length=min_length, lags=lags_int)
        sm_power_sadf = get_sadf(log_prices, model='sm_power', add_const=True, min_length=min_length, lags=lags_int)
        sm_exp_sadf = get_sadf(log_prices, model='sm_exp', add_const=True, min_length=min_length, lags=lags_int)

        sm_power_sadf_phi = get_sadf(log_prices, model='sm_power', add_const=True, min_length=min_length, lags=lags_int,
                                     phi=0.5)
        sm_exp_sadf_phi = get_sadf(log_prices, model='sm_exp', add_const=True, min_length=min_length, lags=lags_int,
                                   phi=0.5)

        self.assertEqual(log_prices.shape[0] - min_length - lags_int - 1, sm_power_sadf.shape[0])  # -1 for series_diff
        self.assertEqual(log_prices.shape[0] - min_length - lags_int - 1, linear_sadf.shape[0])
        self.assertEqual(log_prices.shape[0] - min_length - lags_int - 1, quadratic_sadf.shape[0])
        self.assertEqual(log_prices.shape[0] - min_length - lags_int - 1, sm_poly_1_sadf.shape[0])
        self.assertEqual(log_prices.shape[0] - min_length - lags_int - 1, sm_poly_2_sadf.shape[0])
        self.assertEqual(log_prices.shape[0] - min_length - lags_int - 1, sm_exp_sadf.shape[0])
        self.assertEqual(log_prices.shape[0] - min_length - lags_int - 1, sm_exp_sadf_phi.shape[0])

        self.assertAlmostEqual(sm_power_sadf.mean(), 28.954, delta=1e-3)
        self.assertAlmostEqual(sm_power_sadf.iloc[29], 17.369, delta=1e-3)

        self.assertAlmostEqual(linear_sadf.mean(), -0.669, delta=1e-3)
        self.assertAlmostEqual(linear_sadf[29], -0.717, delta=1e-3)

        self.assertAlmostEqual(linear_sadf_no_const_lags_arr.mean(), 1.899, delta=1e-3)
        self.assertAlmostEqual(linear_sadf_no_const_lags_arr[29], 1.252, delta=1e-3)

        self.assertAlmostEqual(quadratic_sadf.mean(), -1.002, delta=1e-3)
        self.assertAlmostEqual(quadratic_sadf[29], -1.460, delta=1e-3)

        self.assertAlmostEqual(sm_poly_1_sadf.mean(), 26.033, delta=1e-3)
        self.assertAlmostEqual(sm_poly_1_sadf[29], 8.350, delta=1e-3)

        self.assertAlmostEqual(sm_poly_2_sadf.mean(), 26.031, delta=1e-3)
        self.assertAlmostEqual(sm_poly_2_sadf[29], 8.353, delta=1e-3)

        self.assertAlmostEqual(sm_exp_sadf.mean(), 28.916, delta=1e-3)
        self.assertAlmostEqual(sm_exp_sadf[29], 17.100, delta=1e-3)

        self.assertAlmostEqual(sm_power_sadf_phi.mean(), 1.4874, delta=1e-3)
        self.assertAlmostEqual(sm_power_sadf_phi.iloc[29], 2.4564, delta=1e-3)

        self.assertAlmostEqual(sm_exp_sadf_phi.mean(), 1.4787, delta=1e-3)
        self.assertAlmostEqual(sm_exp_sadf_phi[29], 2.4183, delta=1e-3)

        # Trivial series case.
        ones_series = pd.Series(index=log_prices.index, data=np.ones(shape=log_prices.shape[0]))
        trivial_sadf = get_sadf(ones_series, model='sm_power', add_const=True, min_length=min_length, lags=lags_int,
                                phi=0.5)
        self.assertTrue((trivial_sadf.unique() == [-np.inf]).all())  # All values should be -np.inf

        # Test rubbish model argument.
        self.assertRaises(ValueError, get_sadf, series=log_prices, model='rubbish_string', add_const=True,
                          min_length=min_length, lags=lags_int)

        # Assert that nans are parsed if singular matrix
        singular_matrix = np.array([[1, 0, 0], [-1, 3, 3], [1, 2, 2]])
        b_mean, b_var = get_betas(singular_matrix, singular_matrix)
        self.assertTrue(b_mean, [np.nan])
        self.assertTrue(b_var, [[np.nan, np.nan]])
