"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.data_structures import standard_data_structures as ds


class TestDataStructures(unittest.TestCase):
    """
    Test the various financial data structures:
    1. Dollar bars
    2. Volume bars
    3. Tick bars
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/tick_data.csv'

    def test_dollar_bars(self):
        """
        Tests the dollar bars implementation.
        """
        threshold = 100000

        db1 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=1000, verbose=False)
        db2 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False)
        db3 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=10, verbose=False)
        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db4.values == db1.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.5)

        # delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_volume_bars(self):
        """
        Tests the volume bars implementation.
        """
        threshold = 30
        db1 = ds.get_volume_bars(self.path, threshold=threshold, batch_size=1000, verbose=False)
        db2 = ds.get_volume_bars(self.path, threshold=threshold, batch_size=50, verbose=False)
        db3 = ds.get_volume_bars(self.path, threshold=threshold, batch_size=10, verbose=False)
        ds.get_volume_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db4.values == db1.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.75)

        # delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_tick_bars(self):
        """
        Test the tick bars implementation.
        """
        threshold = 10

        db1 = ds.get_tick_bars(self.path, threshold=threshold, batch_size=1000, verbose=False)
        db2 = ds.get_tick_bars(self.path, threshold=threshold, batch_size=50, verbose=False)
        db3 = ds.get_tick_bars(self.path, threshold=threshold, batch_size=10, verbose=False)
        ds.get_tick_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                         to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db4.values == db1.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.50)

        # delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_multiple_csv_file_input(self):
        """
        Tests that bars generated for multiple csv files and Pandas Data Frame yield the same result
        """
        threshold = 100000

        data = pd.read_csv(self.path)

        idx = int(np.round(len(data) / 2))

        data1 = data.iloc[:idx]
        data2 = data.iloc[idx:]

        tick1 = "tick_data_1.csv"
        tick2 = "tick_data_2.csv"

        data1.to_csv(tick1, index=False)
        data2.to_csv(tick2, index=False)

        file_paths = [tick1, tick2]

        db1 = ds.get_dollar_bars(file_paths, threshold=threshold, batch_size=1000, verbose=False)
        db2 = ds.get_dollar_bars(file_paths, threshold=threshold, batch_size=50, verbose=False)
        db3 = ds.get_dollar_bars(file_paths, threshold=threshold, batch_size=10, verbose=False)
        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv', parse_dates=[0])

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db4.values == db1.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.50)

        # delete generated csv files (if they weren't generated test would fail)
        for csv in (tick1, tick2, "test.csv"):
            os.remove(csv)

    def test_df_as_batch_run_input(self):
        """
        Tests that bars generated for csv file and Pandas Data Frame yield the same result
        """
        threshold = 100000
        tick_data = pd.read_csv(self.path)
        tick_data['Date and Time'] = pd.to_datetime(tick_data['Date and Time'])

        db1 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=1000, verbose=False)
        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db2 = pd.read_csv('test.csv')
        db2['date_time'] = pd.to_datetime(db2.date_time)
        db3 = ds.get_dollar_bars(tick_data, threshold=threshold, batch_size=10, verbose=False)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

    def test_list_as_run_input(self):
        """
        Tests that data generated with csv file and list yield the same result
        """
        threshold = 100000
        tick_data = pd.read_csv(self.path)
        tick_data['Date and Time'] = pd.to_datetime(tick_data['Date and Time'])

        db1 = ds.get_dollar_bars(self.path, threshold=threshold, batch_size=1000, verbose=False)
        ds.get_dollar_bars(self.path, threshold=threshold, batch_size=50, verbose=False,
                           to_csv=True, output_path='test.csv')
        db2 = pd.read_csv('test.csv')
        db2['date_time'] = pd.to_datetime(db2.date_time)

        bars = ds.StandardBars(metric='cum_dollar_value', threshold=threshold)
        cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume', 'cum_buy_volume', 'cum_ticks',
                'cum_dollar_value']

        data = tick_data.values.tolist()
        final_bars = bars.run(data)
        db3 = pd.DataFrame(final_bars, columns=cols)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

    def test_wrong_batch_input_value_error_raise(self):
        """
        Tests ValueError raise when neither pd.DataFrame nor path to csv file are passed to function call
        """
        with self.assertRaises(ValueError):
            ds.get_dollar_bars(None, threshold=20, batch_size=1000, verbose=False)

    def test_wrong_run_input_value_error_raise(self):
        """
        Tests ValueError raise when neither pd.DataFrame nor path to csv file are passed to function call
        """
        with self.assertRaises(ValueError):
            bars = ds.StandardBars(metric='cum_dollar_value')
            bars.run(None)

    def test_csv_format(self):
        """
        Asserts that the csv data being passed is of the correct format.
        """
        wrong_date = ['2019-41-30', 200.00, np.int64(5)]
        wrong_price = ['2019-01-30', 'asd', np.int64(5)]
        wrong_volume = ['2019-01-30', 200.00, '1.5']
        too_many_cols = ['2019-01-30', 200.00, np.int64(5), 'Limit order', 'B23']

        # pylint: disable=protected-access
        self.assertRaises(ValueError,
                          ds.StandardBars._assert_csv(pd.DataFrame(wrong_date).T))
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.StandardBars._assert_csv,
                          pd.DataFrame(too_many_cols).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.StandardBars._assert_csv,
                          pd.DataFrame(wrong_price).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.StandardBars._assert_csv,
                          pd.DataFrame(wrong_volume).T)
