"""
Test volatility targeting method.
"""

import os
import unittest

import numpy as np
import pandas as pd

from mlfinlab.bet_sizing.targeting import volatility_targeting


class TestVolatilityTargeting(unittest.TestCase):
    """
    Sets up the data to be used for the following tests.
    """

    def setUp(self):
        project_path = os.path.dirname(__file__)

        # Data set used for testing kelly formula.
        stock_prices = pd.read_csv(project_path + '/test_data/stock_prices.csv', index_col=0)

        spy_df = stock_prices['SPY']
        self.spy_returns = (spy_df / spy_df.shift(1) - 1).dropna()
        self.spy_volatility = self.spy_returns.std()*np.sqrt(252)

        # Default parameters
        self.annualize_factor = 252
        self.target_volatility = 10.

    def test_volatility_targeting(self):
        """
        Tests calcaulating the position size from the volatility targeting.
        """
        position_size = volatility_targeting(self.spy_returns, target_vol=10., annualize_factor=self.annualize_factor)

        expected_volatility = self.spy_volatility*position_size

        # Testing that expected_volatility is adjust to the target volatility
        self.assertAlmostEqual(expected_volatility, self.target_volatility/100., delta=1e-4)
