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
        self.spy_returns = (spy_df / spy_df.shift(1) - 1).dropna()

        assets_df = stock_prices[["SPY", "EEM", "TLT"]]
        self.assets_returns = (assets_df / assets_df.shift(1) - 1).dropna()

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

    def test_kelly_investing(self):
        """
        Tests calculating the optimal leverage from the kelly criterion.
        """
        test_leverage, test_growth_rate = kelly_investing(self.spy_returns,
                                                          risk_free_rate=self.risk_free_rate,
                                                          annualize_factor=self.annualize_factor)

        kelly_leverage, kelly_growth_rate = 0.993942268046857, 0.04359656764182033

        # Testing that value of the leverage amount is right
        self.assertAlmostEqual(test_leverage, kelly_leverage, delta=1e-7)

        # Testing that growth rate of a asset is right
        self.assertAlmostEqual(test_growth_rate, kelly_growth_rate, delta=1e-7)

    def test_kelly_allocation(self):
        """
        Tests calculating the optimal allocations from the kelly criterion.
        """
        test_weights, test_growth_rate = kelly_allocation(self.assets_returns,
                                                          risk_free_rate=self.risk_free_rate,
                                                          annualize_factor=self.annualize_factor)

        kelly_fractions = np.array([5.63723593, -2.67589204, 2.80976013])
        kelly_growth_rate = 0.21340996597420442

        # Testing that weights of portfolio is right
        np.testing.assert_almost_equal(test_weights, kelly_fractions, decimal=7)

        # Testing that growth rate of portfolio is right
        np.testing.assert_almost_equal(test_growth_rate, kelly_growth_rate)
