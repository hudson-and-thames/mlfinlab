# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.fixed_time_horizon import get_forward_return, standardize, fixed_time_horizon


class TestLabellingFixedTime(unittest.TestCase):
    """
    Tests regarding fixed time horizon labelling method
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.data.index = pd.to_datetime(self.data.index)

    def test_get_forward_returns(self):
        """
        Assert that the forward returns for the used test file are accurate
        """
        close = self.data['SPY'][:5]

        # Correct values for the data used
        actual_returns1 = pd.Series([-0.000483, -0.024506, -0.000849, -0.016148, float("NaN")], index=close.index)
        actual_returns3 = pd.Series([-0.025805, -0.041074, float("NaN"), float("NaN"), float("NaN")], index=close.index)

        pd.testing.assert_series_equal(actual_returns1, get_forward_return(close, 1), check_less_precise=3)
        pd.testing.assert_series_equal(actual_returns3, get_forward_return(close, 3), check_less_precise=3)

    def test_standardize(self):
        """
        Assert that standardization is accurate
        """
        # Find forward return
        close = self.data['SPY'][:5]
        returns = get_forward_return(close, 1)

        # Scale by subtracting the mean and dividing by the standard deviation
        mean_std = [(0, 0.005), (0.01, 0.05), (0.02, 0.0007), (-0.1, 0.5), (0.01, 0.01)]
        calculated = standardize(returns, mean_std)

        actual = pd.Series([-0.0966, -0.69012, -29.7843, 0.1677, float("NaN")], index=calculated.index)
        pd.testing.assert_series_equal(calculated, actual, check_less_precise=3)

    def test_fixed_time_horizon(self):
        """
        Assert that the fixed time horizon labelling works as expected.
        Checks a range of static and dynamic thresholds.
        """
        close = self.data['SPY'][:5]

        # Static threshold, no standardization
        for threshold in [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
            for lookfwd in [1, 3]:
                forward_ret = get_forward_return(close, lookfwd)
                labels = fixed_time_horizon(close, threshold, lookfwd=lookfwd)
                for i, _ in enumerate(labels):
                    if labels[i] == 1:
                        self.assertTrue(forward_ret[i] > threshold)
                    if labels[i] == 0:
                        self.assertTrue(np.abs(forward_ret[i]) <= threshold)
                    if labels[i] == -1:
                        self.assertTrue(forward_ret[i] < -threshold)

        # Dynamic threshold, also apply a standardization to the data
        mean_std = [(-0.1, 0.005), (0.01, 0.05), (0.02, 0.0007), (-0.1, 0.5), (0.01, 0.01)]
        threshold_dynamic = pd.Series(np.random.random(len(close)))
        for lookfwd in [1, 3]:
            # Apply standardization to the forward return
            forward_ret = standardize(get_forward_return(close, lookfwd), mean_std)
            labels = fixed_time_horizon(close, threshold_dynamic, lookfwd=lookfwd, standardized=mean_std)
            for i, _ in enumerate(labels):
                if labels[i] == 1:
                    self.assertTrue(forward_ret[i] > threshold_dynamic[i])
                if labels[i] == 0:
                    self.assertTrue(np.abs(forward_ret[i]) <= threshold_dynamic[i])
                if labels[i] == -1:
                    self.assertTrue(forward_ret[i] < -threshold_dynamic[i])

        # Check value error
        with self.assertRaises(ValueError):
            fixed_time_horizon(close, 'string', lookfwd=1)


if __name__ == '__main__':
    unittest.main()
