"""
Test various functions regarding datasets load.
"""

import unittest
from mlfinlab.datasets.load_datasets import (load_tick_sample, load_stock_prices, load_dollar_bar_sample)


class TestDatasets(unittest.TestCase):
    """
    Test load_tick_sample, load_stock_prices, load_dollar_bar_sample.
    """

    def test_load_tick_sample(self):
        """
        Test load_tick_sample function.
        """

        tick_sample_df = load_tick_sample()
        self.assertEqual(tick_sample_df.shape[0], 100)
        self.assertTrue('Price' in tick_sample_df.columns)
        self.assertTrue('Volume' in tick_sample_df.columns)

    def test_load_stock_prices(self):
        """
        Test load_stock_prices function.
        """
        stock_prices_df = load_stock_prices()
        self.assertEqual(stock_prices_df.shape[0], 2141)
        for ticker in ['EEM', 'EWG', 'TIP', 'EWJ', 'EFA', 'IEF', 'EWQ', 'EWU', 'XLB', 'XLE', 'XLF', 'LQD', 'XLK', 'XLU',
                       'EPP', 'FXI', 'VGK', 'VPL', 'SPY', 'TLT', 'BND', 'CSJ', 'DIA']:
            self.assertTrue(ticker in stock_prices_df.columns)

    def test_load_dollar_bar_sample(self):
        """
        Test load_dollar_bar_samples.
        """
        dollar_bars_df = load_dollar_bar_sample()
        self.assertEqual(dollar_bars_df.shape[0], 1000)
        for col in ['open', 'high', 'low', 'close', 'cum_vol', 'cum_dollar', 'cum_ticks']:
            self.assertTrue(col in dollar_bars_df.columns)
