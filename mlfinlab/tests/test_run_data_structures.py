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
    1. Run Dollar bars
    2. Run Volume bars
    3. Run Tick bars
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/imbalance_sample_data.csv'

    def test_ema_run_dollar_bars(self):
        """
        Tests the EMA run dollar bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                   analyse_thresholds=True)
        db2, thresh_2 = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                   analyse_thresholds=True)
        db3, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        self.assertEqual(db1.shape, (3, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[0, 'open'], 1306.0)
        self.assertEqual(db1.loc[0, 'high'], 1306.0)
        self.assertEqual(db1.loc[0, 'low'], 1303.00)
        self.assertEqual(db1.loc[0, 'close'], 1305.75)

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[2, 'open'], 1307.25)
        self.assertEqual(db1.loc[2, 'high'], 1307.25)
        self.assertEqual(db1.loc[2, 'low'], 1302.25)
        self.assertEqual(db1.loc[2, 'close'], 1302.25)

        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())
        self.assertTrue((db1.loc[:, 'volume'] >= db1.loc[:, 'cum_buy_volume']).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_ema_run_volume_bars(self):
        """
        Tests the EMA run volume bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                   analyse_thresholds=True)
        db2, thresh_2 = ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                   analyse_thresholds=True)
        db3, _ = ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        self.assertEqual(db1.shape, (3, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[0, 'open'], 1306.0)
        self.assertEqual(db1.loc[0, 'high'], 1306.0)
        self.assertEqual(db1.loc[0, 'low'], 1303.00)
        self.assertEqual(db1.loc[0, 'close'], 1305.75)

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[2, 'open'], 1307.25)
        self.assertEqual(db1.loc[2, 'high'], 1307.25)
        self.assertEqual(db1.loc[2, 'low'], 1302.25)
        self.assertEqual(db1.loc[2, 'close'], 1302.25)

        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())
        self.assertTrue((db1.loc[:, 'volume'] >= db1.loc[:, 'cum_buy_volume']).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_ema_run_tick_bars(self):
        """
        Tests the EMA run tick bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                 expected_imbalance_window=10000,
                                                 num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                 analyse_thresholds=True)
        db2, thresh_2 = ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                 expected_imbalance_window=10000,
                                                 num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                 analyse_thresholds=True)
        db3, _ = ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                          expected_imbalance_window=10000,
                                          num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                 expected_imbalance_window=10000,
                                 num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                 to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        self.assertEqual(db1.shape, (4, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[0, 'open'], 1306.0)
        self.assertEqual(db1.loc[0, 'high'], 1306.0)
        self.assertEqual(db1.loc[0, 'low'], 1303.00)
        self.assertEqual(db1.loc[0, 'close'], 1305.75)

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[2, 'open'], 1307.25)
        self.assertEqual(db1.loc[2, 'high'], 1307.75)
        self.assertEqual(db1.loc[2, 'low'], 1303.5)
        self.assertEqual(db1.loc[2, 'close'], 1304.5)

        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())
        self.assertTrue((db1.loc[:, 'volume'] >= db1.loc[:, 'cum_buy_volume']).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_ema_run_dollar_bars_with_constraints(self):
        """
        Test the EMA Dollar Run bars with expected number of ticks max and min constraints
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3
        exp_num_ticks_constraints = [100, 1000]

        db1, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            exp_num_ticks_constraints=exp_num_ticks_constraints,
                                            num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False)
        db2, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            exp_num_ticks_constraints=exp_num_ticks_constraints,
                                            num_prev_bars=num_prev_bars, batch_size=50, verbose=False)
        db3, _ = ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            exp_num_ticks_constraints=exp_num_ticks_constraints,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_ema_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   exp_num_ticks_constraints=exp_num_ticks_constraints,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        self.assertEqual(db1.shape, (9, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[0, 'open'], 1306.0)
        self.assertEqual(db1.loc[0, 'high'], 1306.0)
        self.assertEqual(db1.loc[0, 'low'], 1303.0)
        self.assertEqual(db1.loc[0, 'close'], 1305.75)

        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())

        # Assert OHLC is correct (some index)
        self.assertEqual(db1.loc[7, 'open'], 1302.5)
        self.assertEqual(db1.loc[7, 'high'], 1304.75)
        self.assertEqual(db1.loc[7, 'low'], 1301.75)
        self.assertEqual(db1.loc[7, 'close'], 1304.5)

        self.assertTrue((db1.loc[:, 'volume'] >= db1.loc[:, 'cum_buy_volume']).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_const_run_dollar_bars(self):
        """
        Tests the Const run dollar bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                     analyse_thresholds=True)
        db2, thresh_2 = ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                     analyse_thresholds=True)
        db3, _ = ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                              expected_imbalance_window=10000,
                                              num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_const_dollar_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     expected_imbalance_window=10000,
                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                     to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        self.assertEqual(db1.shape, (9, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[0, 'open'], 1306.0)
        self.assertEqual(db1.loc[0, 'high'], 1306.0)
        self.assertEqual(db1.loc[0, 'low'], 1303.00)
        self.assertEqual(db1.loc[0, 'close'], 1305.75)

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[2, 'open'], 1306.0)
        self.assertEqual(db1.loc[2, 'high'], 1307.75)
        self.assertEqual(db1.loc[2, 'low'], 1305.75)
        self.assertEqual(db1.loc[2, 'close'], 1307.75)

        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())
        self.assertTrue((db1.loc[:, 'volume'] >= db1.loc[:, 'cum_buy_volume']).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_const_run_volume_bars(self):
        """
        Tests the Const run volume bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                     analyse_thresholds=True)
        db2, thresh_2 = ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                     expected_imbalance_window=10000,
                                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                     analyse_thresholds=True)
        db3, _ = ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                              expected_imbalance_window=10000,
                                              num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_const_volume_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                     expected_imbalance_window=10000,
                                     num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                     to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        self.assertEqual(db1.shape, (9, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[0, 'open'], 1306.0)
        self.assertEqual(db1.loc[0, 'high'], 1306.0)
        self.assertEqual(db1.loc[0, 'low'], 1303.00)
        self.assertEqual(db1.loc[0, 'close'], 1305.75)

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[2, 'open'], 1306.0)
        self.assertEqual(db1.loc[2, 'high'], 1307.75)
        self.assertEqual(db1.loc[2, 'low'], 1305.75)
        self.assertEqual(db1.loc[2, 'close'], 1307.75)

        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())
        self.assertTrue((db1.loc[:, 'volume'] >= db1.loc[:, 'cum_buy_volume']).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_const_run_tick_bars(self):
        """
        Tests the Const run dollar bars implementation.
        """
        exp_num_ticks_init = 1000
        num_prev_bars = 3

        db1, thresh_1 = ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=2e7, verbose=False,
                                                   analyse_thresholds=True)
        db2, thresh_2 = ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                                   expected_imbalance_window=10000,
                                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                                   analyse_thresholds=True)
        db3, _ = ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                            expected_imbalance_window=10000,
                                            num_prev_bars=num_prev_bars, batch_size=10, verbose=False)
        ds.get_const_tick_run_bars(self.path, exp_num_ticks_init=exp_num_ticks_init,
                                   expected_imbalance_window=10000,
                                   num_prev_bars=num_prev_bars, batch_size=50, verbose=False,
                                   to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        self.assertEqual(db1.shape, (9, 10))

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        self.assertTrue(np.all(thresh_1.cum_theta_buy == thresh_2.cum_theta_buy))
        self.assertTrue(np.all(thresh_1.cum_theta_sell == thresh_2.cum_theta_sell))

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[0, 'open'], 1306.0)
        self.assertEqual(db1.loc[0, 'high'], 1306.0)
        self.assertEqual(db1.loc[0, 'low'], 1303.00)
        self.assertEqual(db1.loc[0, 'close'], 1305.75)

        # Assert OHLC is correct (the first value)
        self.assertEqual(db1.loc[2, 'open'], 1306.0)
        self.assertEqual(db1.loc[2, 'high'], 1307.5)
        self.assertEqual(db1.loc[2, 'low'], 1305.75)
        self.assertEqual(db1.loc[2, 'close'], 1307.5)

        self.assertTrue((db1.loc[:, 'high'] >= db1.loc[:, 'low']).all())
        self.assertTrue((db1.loc[:, 'volume'] >= db1.loc[:, 'cum_buy_volume']).all())

        # Delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_csv_format(self):
        """
        Asserts that the csv data being passed is of the correct format.
        """
        wrong_date = ['2019-41-30', 200.00, np.int64(5)]
        wrong_price = ['2019-01-30', 'asd', np.int64(5)]
        wrong_volume = ['2019-01-30', 200.00, '1.5']
        too_many_cols = ['2019-01-30', 200.00,
                         np.int64(5), 'Limit order', 'B23']

        # pylint: disable=protected-access
        self.assertRaises(ValueError, ds.BaseRunBars._assert_csv(
            pd.DataFrame(wrong_date).T))
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.BaseRunBars._assert_csv,
                          pd.DataFrame(too_many_cols).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.BaseRunBars._assert_csv,
                          pd.DataFrame(wrong_price).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.BaseRunBars._assert_csv,
                          pd.DataFrame(wrong_volume).T)
