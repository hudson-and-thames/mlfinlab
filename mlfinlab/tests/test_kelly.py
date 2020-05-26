"""
Test Kelly Criterion implementations.
"""

import os
import unittest

import numpy as np
import pandas as pd

from mlfinlab.bet_sizing.kelly import kelly_betting, kelly_investing, kelly_allocation


class TestKellyCriterion(unittest.TestCase):
    """
    Sets up the data to be used for the following tests.
    """

    def setUp(self):
        project_path = os.path.dirname(__file__)

        # Data set used for testing kelly formula.
        stock_prices = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0)

        spy_df = stock_prices['SPY']
        spy_log_returns = np.log(spy_df/spy_df.shift(1)).dropna()
        spy_raw_returns = (spy_df / spy_df.shift(1) - 1).dropna()
        self.spy_raw_returns = spy_raw_returns.to_numpy()
        self.spy_log_returns = spy_log_returns.to_numpy()

        assets_df = stock_prices[["SPY", "EEM", "TLT"]]
        assets_log_returns = np.log(assets_df/assets_df.shift(1)).dropna()
        assets_raw_returns = (assets_df / assets_df.shift(1) - 1).dropna()
        self.assets_log_returns = assets_log_returns
        self.assets_raw_returns = assets_raw_returns

        # Default parameters
        self.risk_free_rate = 0.02
        self.annualize_factor = 252

    def test_kelly_betting(self):
        """
        Tests calculating the bet size from the kelly criterion.
        """
        win_probability = 0.53
        profit_unit = 1.0
        loss_unit = 1.0

        test_bet_size, test_growth_rate = kelly_betting(win_probability, profit_unit, loss_unit)
        kelly_bet_size, kelly_growth_rate = 0.06, 0.001801

        # Testing that bet size is right
        self.assertAlmostEqual(test_bet_size, kelly_bet_size, delta=1e-7)

        # Testing that growth rate of the balance is right
        self.assertAlmostEqual(test_growth_rate, kelly_growth_rate, delta=1e-7)

    def test_kelly_investing_log_return(self):
        """
        Tests calculating the optimal leverage from the kelly criterion when return is log return.
        """
        # Test when return is log return
        test_leverage, test_growth_rate = kelly_investing(self.spy_log_returns,
                                                          risk_free_rate=self.risk_free_rate,
                                                          annualize_factor=self.annualize_factor,
                                                          raw_return=False
                                                          )

        kelly_leverage, kelly_growth_rate = 0.4951060602640831, 0.025848714663153166

        # Testing that value of the leverage amount is right
        self.assertAlmostEqual(test_leverage, kelly_leverage, delta=1e-7)

        # Testing that growth rate of a asset is right
        self.assertAlmostEqual(test_growth_rate, kelly_growth_rate, delta=1e-7)

    def test_kelly_investing_raw_return(self):
        """
        Tests calculating the optimal leverage from the kelly criterion when return is raw return.
        """
        # Test when return is raw return
        test_leverage, test_growth_rate = kelly_investing(self.spy_raw_returns,
                                                          risk_free_rate=self.risk_free_rate,
                                                          annualize_factor=self.annualize_factor,
                                                          raw_return=True
                                                          )

        kelly_leverage, kelly_growth_rate = 0.4951060602640831, 0.025848714663153166

        # Testing that value of the leverage amount is right
        self.assertAlmostEqual(test_leverage, kelly_leverage, delta=1e-7)

        # Testing that growth rate of a asset is right
        self.assertAlmostEqual(test_growth_rate, kelly_growth_rate, delta=1e-7)

    def test_kelly_allocation_log_return(self):
        """
        Tests calculating the optimal allocations from the kelly criterion when return is log return.
        """
        test_weights, test_growth_rate = kelly_allocation(self.assets_log_returns,
                                                          risk_free_rate=self.risk_free_rate,
                                                          annualize_factor=self.annualize_factor,
                                                          raw_return=False
                                                          )

        kelly_fractions = np.array([6.17034846, -3.68890982, 1.76385721])
        kelly_growth_rate = 0.22973535914664106

        # Testing that weights of portfolio is right
        np.testing.assert_almost_equal(test_weights, kelly_fractions, decimal=7)

        # Testing that growth rate of portfolio is right
        np.testing.assert_almost_equal(test_growth_rate, kelly_growth_rate)

    def test_kelly_allocation_raw_return(self):
        """
        Tests calculating the optimal allocations from the kelly criterion when return is raw return.
        """
        test_weights, test_growth_rate = kelly_allocation(self.assets_raw_returns,
                                                          risk_free_rate=self.risk_free_rate,
                                                          annualize_factor=self.annualize_factor,
                                                          raw_return=True
                                                          )

        kelly_fractions = np.array([6.17034846, -3.68890982, 1.76385721])
        kelly_growth_rate = 0.22973535914664106

        # Testing that weights of portfolio is right
        np.testing.assert_almost_equal(test_weights, kelly_fractions, decimal=7)

        # Testing that growth rate of portfolio is right
        np.testing.assert_almost_equal(test_growth_rate, kelly_growth_rate)
