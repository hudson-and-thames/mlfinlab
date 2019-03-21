"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.data_structures import run_data_structures as ds


class TestDataStructures(unittest.TestCase):
    """
    Test the various financial data structures:
    1. Imbalance Dollar bars
    2. Imbalance Volume bars
    3. Imbalance Tick bars
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/imbalance_sample_data.csv'

    def test_run_dollar_bars(self):
        """
        Tests the imbalance dollar bars implementation.
        """
        exp_num_ticks_init = 10000
        num_prev_bars = 3
        num_ticks_ewma_window = 10

        db1 = ds.get_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     num_prev_bars=num_prev_bars,
                                     num_ticks_ewma_window=num_ticks_ewma_window, batch_size=1000)
        db2 = ds.get_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     num_prev_bars=num_prev_bars,
                                     num_ticks_ewma_window=num_ticks_ewma_window, batch_size=50)
        db3 = ds.get_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     num_prev_bars=num_prev_bars,
                                     num_ticks_ewma_window=num_ticks_ewma_window, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1306.0)
        self.assertTrue(db1.loc[0, 'high'] == 1308.75)
        self.assertTrue(db1.loc[0, 'low'] == 1301.75)
        self.assertTrue(db1.loc[0, 'close'] == 1304.0)
        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())

    def test_run_volume_bars(self):
        """
        Tests the imbalance volume bars implementation.
        """
        exp_num_ticks_init = 10000
        num_prev_bars = 3
        num_ticks_ewma_window = 10

        db1 = ds.get_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     num_prev_bars=num_prev_bars,
                                     num_ticks_ewma_window=num_ticks_ewma_window, batch_size=1000)
        db2 = ds.get_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     num_prev_bars=num_prev_bars,
                                     num_ticks_ewma_window=num_ticks_ewma_window, batch_size=50)
        db3 = ds.get_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     num_prev_bars=num_prev_bars,
                                     num_ticks_ewma_window=num_ticks_ewma_window, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1306)
        self.assertTrue(db1.loc[0, 'high'] == 1308.75)
        self.assertTrue(db1.loc[0, 'low'] == 1301.75)
        self.assertTrue(db1.loc[0, 'close'] == 1304.0)
        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())

    def test_run_tick_bars(self):
        """
        Tests the imbalance tick bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3
        num_ticks_ewma_window = 10

        db1 = ds.get_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   num_prev_bars=num_prev_bars,
                                   num_ticks_ewma_window=num_ticks_ewma_window, batch_size=1000)
        db2 = ds.get_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   num_prev_bars=num_prev_bars,
                                   num_ticks_ewma_window=num_ticks_ewma_window, batch_size=50)
        db3 = ds.get_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   num_prev_bars=num_prev_bars,
                                   num_ticks_ewma_window=num_ticks_ewma_window, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1306)
        self.assertTrue(db1.loc[0, 'high'] == 1306.00)
        self.assertTrue(db1.loc[0, 'low'] == 1303.5)
        self.assertTrue(db1.loc[0, 'close'] == 1304.25)
        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())

    def test_csv_format(self):
        """
        Asserts that the csv data being passed is of the correct format.
        """
        wrong_date = ['2019-41-30', 200.00, np.int64(5)]
        wrong_price = ['2019-01-30', 'asd', np.int64(5)]
        wrong_volume = ['2019-01-30', 200.00, 1.5]
        too_many_cols = ['2019-01-30', 200.00,
                         np.int64(5), 'Limit order', 'B23']

        # pylint: disable=protected-access
        self.assertRaises(ValueError,
                          ds._assert_dataframe(pd.DataFrame(wrong_date).T))
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds._assert_dataframe,
                          pd.DataFrame(too_many_cols).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds._assert_dataframe,
                          pd.DataFrame(wrong_price).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds._assert_dataframe,
                          pd.DataFrame(wrong_volume).T)
