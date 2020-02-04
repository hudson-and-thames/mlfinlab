"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.data_structures import time_data_structures as ds


class TestTimeDataStructures(unittest.TestCase):
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
        self.path = project_path + '/test_data/tick_data_time_bars.csv'

    def test_day_bars(self):
        """
        Tests the seconds bars implementation.
        """

        db1 = ds.get_time_bars(self.path, resolution='D', num_units=1, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='D', num_units=1, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='D', num_units=1, batch_size=10, verbose=False)
        ds.get_time_bars(self.path, resolution='D', num_units=1, batch_size=50, verbose=False,
                         to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 1)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db1.shape == db4.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db1.values == db4.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1200.0)
        self.assertTrue(db1.loc[0, 'high'] == 1249.75)
        self.assertTrue(db1.loc[0, 'low'] == 1200.0)
        self.assertTrue(db1.loc[0, 'close'] == 1249.75)

        # Assert date_time is correct
        self.assertTrue(db1.loc[0, 'date_time'] == pd.Timestamp(2011, 8, 1, 0, 0, 0).timestamp())

        # delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_hour_bars(self):
        """
        Tests the seconds bars implementation.
        """

        db1 = ds.get_time_bars(self.path, resolution='H', num_units=1, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='H', num_units=1, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='H', num_units=1, batch_size=10, verbose=False)
        ds.get_time_bars(self.path, resolution='H', num_units=1, batch_size=50, verbose=False,
                         to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 3)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db4.values == db1.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[1, 'open'] == 1225.0)
        self.assertTrue(db1.loc[1, 'high'] == 1249.75)
        self.assertTrue(db1.loc[1, 'low'] == 1225.0)
        self.assertTrue(db1.loc[1, 'close'] == 1249.75)

        # Assert date_time is correct
        self.assertTrue(db1.loc[1, 'date_time'] == pd.Timestamp(2011, 8, 1, 0, 0, 0).timestamp())

        # delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_minute_bars(self):
        """
        Tests the minute bars implementation.
        """

        db1 = ds.get_time_bars(self.path, resolution='MIN', num_units=1, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='MIN', num_units=1, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='MIN', num_units=1, batch_size=10, verbose=False)
        ds.get_time_bars(self.path, resolution='MIN', num_units=1, batch_size=50, verbose=False,
                         to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 11)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db4.values == db1.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[9, 'open'] == 1275.0)
        self.assertTrue(db1.loc[9, 'high'] == 1277.0)
        self.assertTrue(db1.loc[9, 'low'] == 1275.0)
        self.assertTrue(db1.loc[9, 'close'] == 1277.0)

        # Assert date_time is correct
        self.assertTrue(db1.loc[9, 'date_time'] == pd.Timestamp(2011, 8, 1, 23, 39, 0).timestamp())

        # delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_second_bars(self):
        """
        Tests the seconds bars implementation.
        """

        db1 = ds.get_time_bars(self.path, resolution='S', num_units=10, batch_size=1000, verbose=False)
        db2 = ds.get_time_bars(self.path, resolution='S', num_units=10, batch_size=50, verbose=False)
        db3 = ds.get_time_bars(self.path, resolution='S', num_units=10, batch_size=10, verbose=False)
        ds.get_time_bars(self.path, resolution='S', num_units=10, batch_size=50, verbose=False,
                         to_csv=True, output_path='test.csv')
        db4 = pd.read_csv('test.csv')

        # Assert diff batch sizes have same number of bars
        self.assertEqual(db1.shape[0], 47)
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)
        self.assertTrue(db4.shape == db1.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))
        self.assertTrue(np.all(db4.values == db1.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[1, 'open'] == 1201.0)
        self.assertTrue(db1.loc[1, 'high'] == 1202.0)
        self.assertTrue(db1.loc[1, 'low'] == 1201.0)
        self.assertTrue(db1.loc[1, 'close'] == 1202.0)

        # Assert date_time is correct
        self.assertTrue(db1.loc[1, 'date_time'] == pd.Timestamp(2011, 7, 31, 22, 39, 0).timestamp())

        # delete generated csv file (if it wasn't generated test would fail)
        os.remove('test.csv')

    def test_wrong_input_value_error_raise(self):
        """
        Tests ValueError raise when neither pd.DataFrame nor path to csv file are passed to function call
        """
        with self.assertRaises(ValueError):
            ds.get_time_bars(None, resolution='MIN', num_units=1, batch_size=1000, verbose=False)

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
                          ds.TimeBars._assert_csv(pd.DataFrame(wrong_date).T))
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.TimeBars._assert_csv,
                          pd.DataFrame(too_many_cols).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.TimeBars._assert_csv,
                          pd.DataFrame(wrong_price).T)
        # pylint: disable=protected-access
        self.assertRaises(AssertionError,
                          ds.TimeBars._assert_csv,
                          pd.DataFrame(wrong_volume).T)
