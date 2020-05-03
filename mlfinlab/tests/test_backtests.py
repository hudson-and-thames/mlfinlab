"""
Tests the backtests of Campbell research - Haircut Sharpe ratio and Profit Hurdle algorithms.
"""
import unittest
import numpy as np
from mlfinlab.backtest_statistics.backtests import CampbellBacktesting


class TestCampbellBacktesting(unittest.TestCase):
    # pylint: disable=protected-access
    """
    Tests functions of the CampbellBacktesting class.
    """

    def test_haircut_sharpe_ratios_simple_input(self):
        """
        Test the calculation of haircuts with simple inputs
        """

        sample_frequency = 'M'
        num_observations = 120
        sharpe_ratio = 1
        annualized = True
        autocorr_adjusted = False
        rho_a = 0.1
        num_mult_test = 100
        rho = 0.4
        parameters = (sample_frequency, num_observations, sharpe_ratio, annualized,
                      autocorr_adjusted, rho_a, num_mult_test, rho)

        # Avoiding a random output
        np.random.seed(0)

        backtesting = CampbellBacktesting(400)
        haircuts = backtesting.haircut_sharpe_ratios(*parameters)

        # Testing the adjusted p-values as other outputs are calculated from those
        self.assertAlmostEqual(haircuts[0][0], 0.465, delta=1e-2)
        self.assertAlmostEqual(haircuts[0][1], 0.409, delta=1e-2)
        self.assertAlmostEqual(haircuts[0][2], 0.174, delta=1e-2)
        self.assertAlmostEqual(haircuts[0][3], 0.348, delta=1e-2)

        print(haircuts[0][2])

    def test_profit_hurdle(self):
        """
        Test the calculation of haircuts with simple inputs
        """

        num_mult_test = 300
        num_obs = 240
        alpha_sig = 0.05
        vol_anu = 0.1
        rho = 0.4
        parameters = (num_mult_test, num_obs, alpha_sig, vol_anu, rho)

        # Avoiding a random output
        np.random.seed(0)

        backtesting = CampbellBacktesting(400)
        p_values = backtesting.profit_hurdle(*parameters)

        # Testing the adjusted p-values as other outputs are calculated from them
        self.assertAlmostEqual(p_values[0], 0.365, delta=1e-2)
        self.assertAlmostEqual(p_values[1], 0.702, delta=1e-2)
        self.assertAlmostEqual(p_values[2], 0.687, delta=1e-2)
        self.assertAlmostEqual(p_values[3], 0.620, delta=1e-2)
        self.assertAlmostEqual(p_values[4], 0.694, delta=1e-2)

    def test_holm_method_returns(self):
        """
        Test the special inputs to Holm method on required monthly returns.
        Particularly, when there are no exceeding p-values in the simulations
        """

        p_values_simulation = [0.001, 0.0011, 0.0012, 0.0013]
        num_mult_test = 4
        alpha_sig = 0.05
        parameters = (p_values_simulation, num_mult_test, alpha_sig)

        # Avoiding a random output
        np.random.seed(0)

        backtesting = CampbellBacktesting(200)
        tstat = backtesting._holm_method_returns(*parameters)

        # Testing the resulting t-statistic
        self.assertEqual(tstat, 1.96)


    def test_bhy_method_returns(self):
        """
        Test the special inputs to BHY method on required monthly returns.
        """

        p_values_simulation = [0.1, 0.11, 0.12, 0.13]
        p_values_simulation_low = [0.001, 0.0011, 0.0012, 0.0013]
        num_mult_test = 4
        alpha_sig = 0.05

        # Avoiding a random output
        np.random.seed(0)

        backtesting = CampbellBacktesting(200)

        # When too few multiple tests
        tstat = backtesting._bhy_method_returns(p_values_simulation, 1, alpha_sig)
        self.assertEqual(tstat, 1.96)

        # If no exceeding p-values
        tstat = backtesting._bhy_method_returns(p_values_simulation, num_mult_test, alpha_sig)
        self.assertEqual(tstat, 1.96)

        # If exceeding value is first
        tstat = round(backtesting._bhy_method_returns(p_values_simulation_low, num_mult_test, alpha_sig), 3)
        self.assertEqual(tstat, 3.216)

    def test_parameter_calculation(self):
        """
        Test the calculation of parameters of HLZ model by correlation
        """

        backtesting = CampbellBacktesting(200)
        rho = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
        expected_result = [0.2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.2]
        parameters = []

        # Avoiding a random output
        np.random.seed(0)

        for rho_el in rho:
            parameters.append(round(backtesting._parameter_calculation(rho_el)[0], 2))
        self.assertEqual(expected_result, parameters)

    def test_annualized_sharpe_ratio(self):
        """
        Test the conversion of Sharpe ratio to annualized and adjusted to autocorrelation of returns
        """

        backtesting = CampbellBacktesting(200)
        sampling_frequency = ['D', 'W', 'M', 'Q', 'A', 'N']
        expected_result_autocorr = [0.905, 0.906, 0.912, 0.928, 1.0, 1.0]
        expected_result_annual = [18.974, 7.211, 3.464, 2.0, 1.0, 1.0]
        parameters_autocorr = []

        # Avoiding a random output
        np.random.seed(0)

        # Tests for not adjusted to autocorrelation
        for freq in sampling_frequency:
            sr_annual_adj = backtesting._annualized_sharpe_ratio(sharpe_ratio=1, sampling_frequency=freq, rho=0.1,
                                                                 annualized=True, autocorr_adjusted=False)
            parameters_autocorr.append(round(sr_annual_adj, 3))
        self.assertEqual(expected_result_autocorr, parameters_autocorr)

        parameters_annual = []
        # Test for already adjusted
        for freq in sampling_frequency:
            sr_annual_adj = backtesting._annualized_sharpe_ratio(sharpe_ratio=1, sampling_frequency=freq, rho=0.1,
                                                                 annualized=False, autocorr_adjusted=True)
            parameters_annual.append(round(sr_annual_adj, 3))
        self.assertEqual(expected_result_annual, parameters_annual)

    def test_monthly_observations(self):
        """
        Test the calculation of monthly observations in a sample
        """

        backtesting = CampbellBacktesting(200)
        sampling_frequency = ['D', 'W', 'M', 'Q', 'A', 'N']
        expected_observations = [0.0, 2.0, 10.0, 30.0, 120., 10.0]
        observations = []

        # Avoiding a random output
        np.random.seed(0)

        # Tests for not adjusted to autocorrelation
        for freq in sampling_frequency:
            monthly_obs = backtesting._monthly_observations(10, freq)
            observations.append(monthly_obs)
        self.assertEqual(expected_observations, observations)
