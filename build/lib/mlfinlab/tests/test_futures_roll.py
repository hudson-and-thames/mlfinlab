"""
Tests the financial data structures
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.multi_product.etf_trick import get_futures_roll_series


class TestETFTrick(unittest.TestCase):
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
        path = project_path + '/test_data'

        self.open_df_path = '{}/open_df.csv'.format(path)
        self.close_df_path = '{}/close_df.csv'.format(path)

        self.open_df = pd.read_csv(self.open_df_path, usecols=['date', 'spx'])
        self.close_df = pd.read_csv(self.close_df_path, usecols=['date', 'spx'])

        self.open_df.rename(columns={'spx': 'open'}, inplace=True)
        self.close_df.rename(columns={'spx': 'close'}, inplace=True)

    def test_futures_roll(self):
        """
        Tests get_futures_roll function
        """
        combined_df = self.open_df.merge(self.close_df, on='date')
        combined_df['date'] = pd.to_datetime(combined_df.date)
        combined_df.set_index('date', inplace=True)
        roll_dates = {'futures_1': pd.Timestamp(2017, 3, 20),
                      'futures_2': pd.Timestamp(2018, 1, 17)}  # futures roll dates

        combined_df['current_futures'] = np.where(combined_df.index <= roll_dates['futures_1'], 'futures_1',
                                                  np.where(combined_df.index <= roll_dates['futures_2'], 'futures_2',
                                                           'futures_3'))

        res = get_futures_roll_series(combined_df, 'open', 'close', 'current_futures', 'current_futures')
        combined_df.loc[res.index, 'roll_gap'] = res.values
        combined_df['open_diff_close'] = combined_df.open - combined_df.close

        # Test number of gaps (should be 2 => number of unique gaps should be 3 (0 added)
        self.assertTrue(len(combined_df['roll_gap'].unique()) == len(roll_dates) + 1)  # zero value + rolling gaps

        res_backward = get_futures_roll_series(combined_df, 'open', 'close', 'current_futures', 'current_futures',
                                               roll_backward=True)

        # Test difference between backward and forward roll equal latest cumulative gap
        self.assertTrue(np.all(res.iloc[-1] == 33.75))
        self.assertTrue(np.all(res_backward.iloc[0] == -33.75))
        self.assertTrue(np.all(res.unique() == [0, 26.75, 33.75]))
