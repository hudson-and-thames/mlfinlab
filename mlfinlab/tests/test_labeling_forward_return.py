import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.forward_return import forward_return


class TestLabelingForwardReturn(unittest.TestCase):
    """
    Tests for labeling according to forward return.
    """
    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.idx5 = self.data[:5].index
        self.cols10 = self.data.iloc[:, 0:10].columns

    def test_forward_return(self):
        """
        Test verifying that forward return works correctly on data.
        """
        prices = self.data.iloc[0:5, 0:10]
        test1 = forward_return(prices)
        test1_actual = pd.DataFrame([[1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                     [1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                                     [np.nan]*10], index=self.idx5, columns=self.cols10)
        test2_prices = pd.Series([100, 110, 99, 95, 10, 18, 22, 99, 3000, 0.1,  2])
        test2 = forward_return(test2_prices, lookforward=3)
        test2_actual = pd.Series([0, 0, 0, 0, 1, 1, 0, 0, np.nan, np.nan, np.nan])

        pd.testing.assert_frame_equal(test1, test1_actual)
        pd.testing.assert_series_equal(test2, test2_actual)

    def test_warnings(self):
        """
        Tests that correct warnings and errors are given if user inputs a wrong lookforward period.
        """
        prices = self.data.iloc[0:5, 0:10]
        with self.assertWarns(UserWarning):
            forward_return(prices, lookforward=999)
        with self.assertRaises(Exception):
            forward_return(prices, lookforward='str')







