"""
Tests the backtests of Campbell research - Haircut Sharpe ratio and Profit Hurdle algorithms.
"""

import unittest
from mlfinlab.backtest_statistics.backtests import CampbellBacktesting


class TestCampbellBacktesting(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests functions of the CampbellBacktesting class.
    """

    def setUp(self):
        """

        """


    def test_haircut_sharpe_ratio_simple_input(self):
        """
        Test the calculation of haircuts with simple inputs
        """
        sample_frequency = 'M'
        num_observations = 120
        sharpe_ratio = True
        annualized = False
        autocorr_adjusted = 1
        rho_a = 0.1
        num_mult_test = 100
        rho = 0.4
        parameters = (sample_frequency, num_observations, sharpe_ratio, annualized,
                      autocorr_adjusted, rho_a, num_mult_test, rho)

        backtesting = CampbellBacktesting()
        haircuts = backtesting.haircut_sharpe_ratios(*parameters)
        print(haircuts)

        # Testing the adjusted p-values as other outputs are calculated from those
        self.assertAlmostEqual(haircuts[0][0], 0.465, delta=1e-3)
        self.assertAlmostEqual(haircuts[1][0], 0.409, delta=1e-3)
        self.assertAlmostEqual(haircuts[2][0], 0.170, delta=1e-3)
        self.assertAlmostEqual(haircuts[3][0], 0.348, delta=1e-3)