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

        gaps_diff_no_backward = get_futures_roll_series(combined_df, 'open', 'close', 'current_futures',
                                                        'current_futures', method='absolute', roll_backward=False)
        gaps_rel_no_backward = get_futures_roll_series(combined_df, 'open', 'close', 'current_futures',
                                                       'current_futures', method='relative', roll_backward=False)
        gaps_diff_with_backward = get_futures_roll_series(combined_df, 'open', 'close', 'current_futures',
                                                          'current_futures', method='absolute', roll_backward=True)
        gaps_rel_with_backward = get_futures_roll_series(combined_df, 'open', 'close', 'current_futures',
                                                         'current_futures', method='relative', roll_backward=True)
        with self.assertRaises(ValueError):
            get_futures_roll_series(combined_df, 'open', 'close', 'current_futures',
                                    'current_futures', method='unknown', roll_backward=True)

        # Test number of gaps (should be 2 => number of unique gaps should be 3 (0 added)
        self.assertTrue(len(gaps_diff_no_backward.unique()) == len(roll_dates) + 1)
        self.assertTrue(len(gaps_rel_no_backward.unique()) == len(roll_dates) + 1)

        # Assert values of difference method Futures Roll Trick
        self.assertTrue(gaps_diff_no_backward.iloc[0] == 0)
        self.assertTrue(gaps_diff_no_backward.iloc[-1] == -1.75)
        self.assertTrue(gaps_diff_with_backward.iloc[0] == 1.75)
        self.assertTrue(gaps_diff_with_backward.iloc[-1] == 0)

        # Assert values of relative method Futures Roll Trick
        self.assertTrue(gaps_rel_no_backward.iloc[0] == 1)
        self.assertTrue(abs(gaps_rel_no_backward.iloc[-1] - 0.999294) < 1e-6)
        self.assertTrue(abs(gaps_rel_with_backward.iloc[0] - 1 / 0.999294) < 1e-6)
        self.assertTrue(gaps_rel_with_backward.iloc[-1] == 1)
