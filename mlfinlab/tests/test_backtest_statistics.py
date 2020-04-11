"""
Test various functions related to Backtest Statistics
"""
import os
import unittest

import datetime as dt
import pandas as pd
import numpy as np

from mlfinlab.backtest_statistics.statistics import (timing_of_flattening_and_flips, average_holding_period,
                                                     bets_concentration, all_bets_concentration,
                                                     drawdown_and_time_under_water, sharpe_ratio,
                                                     information_ratio, probabilistic_sharpe_ratio,
                                                     deflated_sharpe_ratio, minimum_track_record_length)


class TestBacktestStatistics(unittest.TestCase):
    """
    Test following functions in statistocs.py:
    - timing_of_flattening_and_flips
    - average_holding_period
    - bets_concentration
    - all_bets_concentration
    - compute_drawdown_and_time_under_water
    - sharpe_ratio
    - information ratio
    - probabilistic_sharpe_ratio
    - deflated_sharpe_ratio
    - minimum_track_record_length
    """

    def setUp(self):
        """
        Set the data for tests.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/dollar_bar_sample.csv'
        self.logret = pd.read_csv(data_path, index_col='date_time')
        self.logret.index = pd.to_datetime(self.logret.index)
        self.logret = np.log(self.logret['close']).diff()[1:]

        dates = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(10)])
        flip_positions = np.array([1.0, 1.5, 0.5, 0, -0.5, -1.0, 0.5, 1.5, 1.5, 1.5])
        hold_positions = np.array([0, 1, 1, -1, -1, 0, 0, 2, 2, 0])
        no_closed_positions = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        dollar_ret = np.array([100, 110, 90, 100, 120, 130, 100, 120, 140, 130])
        normal_ret = np.array([0.01, 0.03, 0.02, 0.01, -0.01, 0.02, 0.01, 0.0, -0.01, 0.01])
        cumulated_ret = np.cumprod(1 + normal_ret)

        self.flip_flattening_positions = pd.Series(data=flip_positions, index=dates)
        self.flips = pd.DatetimeIndex([dt.datetime(2000, 1, 7)])
        self.flattenings = pd.DatetimeIndex([dt.datetime(2000, 1, 4), dt.datetime(2000, 1, 10)])
        self.hold_positions = pd.Series(data=hold_positions, index=dates)
        self.no_closed_positions = pd.Series(data=no_closed_positions, index=dates)
        self.dollar_returns = pd.Series(data=dollar_ret, index=dates)
        # Dropping first observation as it's being lost in conversion
        self.normal_returns = pd.Series(data=normal_ret, index=dates)[1:]
        self.cumulated_returns = pd.Series(data=cumulated_ret, index=dates)

    def test_timing_of_flattening_and_flips(self):
        """
        Check that moments of flips and flattenings are picked correctly and
        that last is added
        """

        flattenings_and_flips = timing_of_flattening_and_flips(self.flip_flattening_positions)
        test_flat_flip = self.flips.append(self.flattenings)

        # In case last bet is already included
        altered_flips = self.flip_flattening_positions.copy()
        altered_flips[-1:] = 0
        flattenings_and_flips_last = timing_of_flattening_and_flips(altered_flips)

        self.assertTrue(test_flat_flip.sort_values().equals(flattenings_and_flips.sort_values()))
        self.assertTrue(flattenings_and_flips_last.sort_values().equals(flattenings_and_flips.sort_values()))

    def test_average_holding_period(self):
        """
        Check average holding period calculation
        """

        average_holding = average_holding_period(self.hold_positions)
        nan_average_holding = average_holding_period(self.no_closed_positions)

        # As seen from example set, positions are kept 2 days on average
        self.assertAlmostEqual(average_holding, 2, delta=1e-4)
        self.assertTrue(np.isnan(nan_average_holding))

    def test_bets_concentration(self):
        """
        Check if concentration is balanced and correctly calculated
        """

        positive_concentration = bets_concentration(self.logret)
        #Testing for symmetry in concentration
        flipped_logret = -1 * self.logret
        negative_concentration = bets_concentration(flipped_logret)

        self.assertAlmostEqual(positive_concentration, negative_concentration,
                               delta=1e-5)
        self.assertAlmostEqual(positive_concentration, 2.0111445, delta=1e-4)

    def test_all_bets_concentration(self):
        """
        Check if concentration is nan when not enough observations, also values
        testing
        """

        # Only one negative return in this dataset
        positive_returns_concentration = all_bets_concentration(self.normal_returns)

        # A longer dataset with all concentrations
        all_returns_concentration = all_bets_concentration(self.logret, frequency='D')

        # Not enough negative observations in dataset
        self.assertTrue(np.isnan(positive_returns_concentration[1]))

        # Not enough observations to group by month
        self.assertTrue(np.isnan(positive_returns_concentration[2]))
        self.assertAlmostEqual(all_returns_concentration[0], 0.0014938,
                               delta=1e-5)
        self.assertAlmostEqual(all_returns_concentration[1], 0.0016261,
                               delta=1e-5)
        self.assertAlmostEqual(all_returns_concentration[2], 0.0195998,
                               delta=1e-5)

    def test_drawdown_and_time_under_water(self):
        """
        Check if drawdowns and time under water calculated correctly for
        dollar and non-dollar test sets.
        """

        drawdown_dol, time_under_water_dol = drawdown_and_time_under_water(self.dollar_returns,
                                                                           dollars=True)
        _, time_under_water = drawdown_and_time_under_water(self.dollar_returns / 100,
                                                            dollars=False)

        self.assertTrue(list(drawdown_dol) == [20.0, 30.0, 10.0])
        self.assertTrue(list(time_under_water) == list(time_under_water_dol))
        self.assertAlmostEqual(time_under_water[0], 0.010951,
                               delta=1e-4)
        self.assertAlmostEqual(time_under_water[1], 0.008213,
                               delta=1e-4)

    def test_sharpe_ratio(self):
        """
        Check if Sharpe ratio is calculated right
        """

        sharpe = sharpe_ratio(self.normal_returns, entries_per_year=12,
                              risk_free_rate=0.005)

        self.assertAlmostEqual(sharpe, 0.987483, delta=1e-4)

    def test_information_ratio(self):
        """
        Check if Information ratio is calculated right
        """

        information_r = information_ratio(self.normal_returns, benchmark=0.006,
                                          entries_per_year=12)

        self.assertAlmostEqual(information_r, 0.733559, delta=1e-4)

    def test_probabilistic_sharpe_ratio(self):
        """
        Check probabilistic Sharpe ratio using numerical example
        """

        observed_sr = 1.14
        benchmark_sr = 1
        number_of_returns = 250
        skewness = 0
        kurtosis = 3

        result_prob_sr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr,
                                                    number_of_returns, skewness,
                                                    kurtosis)

        self.assertAlmostEqual(result_prob_sr, 0.95727,
                               delta=1e-4)

    def test_deflated_sharpe_ratio(self):
        """
        Check deflated Sharpe ratio using numerical example
        """

        observed_sr = 1.14
        sr_estimates = [3.5, 1.01, 1.02]

        # Parameters of SR estimates - standard deviation and number of observations
        estim_param = [0.4, 100]
        number_of_returns = 250
        skewness = 0
        kurtosis = 3

        result_defl_sr = deflated_sharpe_ratio(observed_sr, sr_estimates,
                                               number_of_returns, skewness,
                                               kurtosis)

        benchmark_sr = deflated_sharpe_ratio(observed_sr, estim_param,
                                             number_of_returns, skewness,
                                             kurtosis, estimates_param=True,
                                             benchmark_out=True)

        param_defl_sr = deflated_sharpe_ratio(observed_sr, estim_param,
                                              number_of_returns, skewness,
                                              kurtosis, estimates_param=True)


        self.assertAlmostEqual(result_defl_sr, 0.95836,
                               delta=1e-4)

        self.assertAlmostEqual(benchmark_sr, 1.012241,
                               delta=1e-4)

        self.assertAlmostEqual(param_defl_sr, 0.941740,
                               delta=1e-4)

    def test_minimum_track_record_length(self):
        """
        Check deflated Sharpe ratio using numerical example
        """

        observed_sr = 1.14
        benchmark_sr = 1
        skewness = 0
        kurtosis = 3
        alpha = 0.05

        result_min_track_rec = minimum_track_record_length(observed_sr,
                                                           benchmark_sr,
                                                           skewness,
                                                           kurtosis,
                                                           alpha)

        self.assertAlmostEqual(result_min_track_rec, 228.73497,
                               delta=1e-4)
