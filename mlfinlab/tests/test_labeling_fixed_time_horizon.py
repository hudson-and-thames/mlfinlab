import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling import fixed_time_horizon


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
        self.data = pd.read_csv(self.path, index_col='date_time')
        self.data.index = pd.to_datetime(self.data.index)

    def test_fixed_time_horizon(self):
        """
        Assert that the fixed time horizon labelling works as expected.
        Checks a range of static and dynamic thresholds.
        """
        close = self.data['SPY']
        daily_ret = close.pct_change(periods=h)
        forward_ret = pd.Series(list(daily_ret)[h:] + [float("NaN")] * h, index=close.index)

        # static threshold
        for threshold in [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
            for h in [1, 3]:
                labels = fixed_time_horizon(close, threshold, h=h)
                for i in range(len(labels)):
                    if labels[i] == 1:
                        self.assertTrue(forward_ret[i] > threshold)
                    if labels[i] == 0:
                        self.assertTrue(np.abs(forward_ret[i]) <= threshold)
                    if labels[i] == -1:
                        self.assertTrue(forward_ret[i] < -threshold)

        # dynamic threshold
        threshold_d = np.random.random(len(self.data['SPY'] / 50))
        for h in [1, 3]:
            labels = fixed_time_horizon(close, threshold_d, h=h)
            for i in range(len(labels)):
                if labels[i] == 1:
                    self.assertTrue(forward_ret[i] > threshold)
                if labels[i] == 0:
                    self.assertTrue(np.abs(forward_ret[i]) <= threshold)
                if labels[i] == -1:
                    self.assertTrue(forward_ret[i] < -threshold)


if __name__ == '__main__':
    unittest.main()
