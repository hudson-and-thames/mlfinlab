
import unittest
import numpy as np
import pandas as pd

from mlfinlab.data_structures import data_structures as ds


class TestDataStructures(unittest.TestCase):

    def test_dollar_bars(self):
        threshold = 100000

        db1 = ds.get_dollar_bars('./mlfinlab/tests/test_data/tick_data.csv', threshold=threshold, batch_size=100)
        db2 = ds.get_dollar_bars('./mlfinlab/tests/test_data/tick_data.csv', threshold=threshold, batch_size=50)
        db3 = ds.get_dollar_bars('./mlfinlab/tests/test_data/tick_data.csv', threshold=threshold, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.5)

        # Assert cum dollar value greater than threshold
        self.assertTrue(np.all(db1['cum_dollar'] >= threshold))

    def test_volume_bars(self):
        threshold = 30

        db1 = ds.get_dollar_bars('./mlfinlab/tests/test_data/tick_data.csv', threshold=threshold, batch_size=100)
        db2 = ds.get_dollar_bars('./mlfinlab/tests/test_data/tick_data.csv', threshold=threshold, batch_size=50)
        db3 = ds.get_dollar_bars('./mlfinlab/tests/test_data/tick_data.csv', threshold=threshold, batch_size=10)

        # Assert diff batch sizes have same number of bars
        self.assertTrue(db1.shape == db2.shape)
        self.assertTrue(db1.shape == db3.shape)

        # Assert same values
        self.assertTrue(np.all(db1.values == db2.values))
        self.assertTrue(np.all(db1.values == db3.values))

        # Assert OHLC is correct
        self.assertTrue(db1.loc[0, 'open'] == 1205)
        self.assertTrue(db1.loc[0, 'high'] == 1904.75)
        self.assertTrue(db1.loc[0, 'low'] == 1005.0)
        self.assertTrue(db1.loc[0, 'close'] == 1304.5)

        # Assert cum dollar value greater than threshold
        self.assertTrue(np.all(db1['cum_volume'] >= threshold))
