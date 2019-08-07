"""
Tests the higher level functions in bet_sizing.py.
"""

import unittest
from unittest.mock import patch
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, moment

from mlfinlab.bet_sizing.bet_sizing import (bet_size_probability, bet_size_dynamic, bet_size_budget, bet_size_reserve,
                                            confirm_and_cast_to_df, get_concurrent_sides, cdf_mixture,
                                            single_bet_size_mixed)
from mlfinlab.bet_sizing.ch10_snippets import (get_signal, avg_active_signals, discrete_signal,
                                               get_w, get_target_pos, limit_price, bet_size)
from mlfinlab.bet_sizing.ef3m import M2N, raw_moment, most_likely_parameters

class TestBetSizeProbability(unittest.TestCase):
    """
    Tests the 'bet_size_probability' function.
    """
    def test_bet_size_probability_default(self):
        """
        Tests for successful execution using the default arguments of 'bet_size_probability', which are:
         average_active = False
         step_size = 0.0
        """
        # Setup the test DataFrame.
        dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
        shift_dt = np.array([dt.timedelta(days=0.5*i+1) for i in range(5)])
        dates_shifted_test = dates_test + shift_dt
        events_test = pd.DataFrame(data=[[0.55, 1], [0.7, -1], [0.95, 1], [0.65, -1], [0.85, 1]],
                                   columns=['prob', 'side'],
                                   index=dates_test)
        events_test['t1'] = dates_shifted_test
        # Calculate correct output.
        signal_0 = get_signal(events_test['prob'], 2, events_test['side'])
        df_signal_0 = signal_0.to_frame('signal').join(events_test['t1'], how='left')
        signal_1 = df_signal_0.signal
        # Evaluate test.
        self.assertTrue(signal_1.equals(bet_size_probability(events_test, events_test['prob'], 2, events_test['side'])))

    def test_bet_size_probability_avg_active(self):
        """
        Tests for successful execution of 'bet_size_probability' with 'average_active' set to True.
        """
        # Setup the test DataFrame.
        dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
        shift_dt = np.array([dt.timedelta(days=0.5*i+1) for i in range(5)])
        dates_shifted_test = dates_test + shift_dt
        events_test = pd.DataFrame(data=[[0.55, 1], [0.7, -1], [0.95, 1], [0.65, -1], [0.85, 1]],
                                   columns=['prob', 'side'],
                                   index=dates_test)
        events_test['t1'] = dates_shifted_test
        # Calculate correct output.
        signal_0 = get_signal(events_test['prob'], 2, events_test['side'])
        df_signal_0 = signal_0.to_frame('signal').join(events_test['t1'], how='left')
        signal_1 = avg_active_signals(df_signal_0, 1)
        # Evaluate test.
        self.assertTrue(signal_1.equals(bet_size_probability(events=events_test, prob=events_test['prob'], num_classes=2,
                                                             pred=events_test['side'], average_active=True)))

    def test_bet_size_probability_stepsize(self):
        """
        Tests for successful execution of 'bet_size_probability' with 'step_size' greater than 0.
        """
        # Setup the test DataFrame.
        dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
        shift_dt = np.array([dt.timedelta(days=0.5*i+1) for i in range(5)])
        dates_shifted_test = dates_test + shift_dt
        events_test = pd.DataFrame(data=[[0.55, 1], [0.7, -1], [0.95, 1], [0.65, -1], [0.85, 1]],
                                   columns=['prob', 'side'],
                                   index=dates_test)
        events_test['t1'] = dates_shifted_test
        # Calculate correct output.
        signal_0 = get_signal(events_test['prob'], 2, events_test['side'])
        df_signal_0 = signal_0.to_frame('signal').join(events_test['t1'], how='left')
        signal_1 = df_signal_0.signal
        signal_1 = discrete_signal(signal0=signal_1, step_size=0.1)
        # Evaluate test.
        self.assertTrue(signal_1.equals(bet_size_probability(events=events_test, prob=events_test['prob'], num_classes=2,
                                                             pred=events_test['side'], step_size=0.1)))

class TestBetSizeDynamic(unittest.TestCase):
    """
    Tests the 'bet_size_dynamic' function.
    """
    def test_bet_size_dynamic_default(self):
        """
        Tests for successful execution using the default arguments of 'bet_size_dynamic', which are:
         average_active = False
         step_size = 0.0
        """
        # Setup the test DataFrame.
        dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
        events_test = pd.DataFrame(data=[[25, 55, 75.50, 80.00],
                                         [35, 55, 76.90, 75.00],
                                         [45, 55, 74.10, 72.50],
                                         [40, 55, 67.75, 65.00],
                                         [30, 55, 62.00, 70.80]],
                                   columns=['pos', 'max_pos', 'm_p', 'f'],
                                   index=dates_test)
        # Calculate results.
        d_events = {col: events_test[col] for col in list(events_test.columns)}
        events_results = confirm_and_cast_to_df(d_events)
        w_param = get_w(10, 0.95, 'sigmoid')
        events_results['t_pos'] = events_results.apply(lambda row: get_target_pos(w_param, row.f, row.m_p,
                                                                                  row.max_pos, 'sigmoid'), axis=1)
        events_results['l_p'] = events_results.apply(lambda row: limit_price(row.t_pos, row.pos, row.f, w_param,
                                                                             row.max_pos, 'sigmoid'), axis=1)
        events_results['bet_size'] = events_results.apply(lambda row: bet_size(w_param, row.f-row.m_p, 'sigmoid'),
                                                          axis=1)
        df_result = events_results[['bet_size', 't_pos', 'l_p']]
        # Evaluate.
        self.assertTrue(df_result.equals(bet_size_dynamic(events_test['pos'], events_test['max_pos'],
                                                          events_test['m_p'], events_test['f'])))

class TestBetSizeBudget(unittest.TestCase):
    """
    Tests the 'bet_size_budget' function.
    """
    def test_bet_size_budget_default(self):
        """
        Tests for the successful execution of the 'bet_size_budget' function.
        """
        # Setup the test DataFrame.
        dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
        shift_dt = np.array([dt.timedelta(days=0.5*i+1) for i in range(5)])
        dates_shifted_test = dates_test + shift_dt
        events_test = pd.DataFrame(data=[[0.55, 1], [0.7, 1], [0.95, 1], [0.65, -1], [0.85, 1]],
                                   columns=['prob', 'side'],
                                   index=dates_test)
        events_test['t1'] = dates_shifted_test
        # Calculate correct result.
        events_result = get_concurrent_sides(events_test['t1'], events_test['side'])
        avg_long = events_result['active_long'] / events_result['active_long'].max()
        avg_short = events_result['active_short'] / events_result['active_short'].max()
        events_result['bet_size'] = avg_long - avg_short
        # Evaluate.
        self.assertTrue(events_result.equals(bet_size_budget(events_test['t1'], events_test['side'])))

class TestBetSizeReserve(unittest.TestCase):
    """
    Tests the 'bet_size_reserve' function.
    """
    @patch('mlfinlab.bet_sizing.bet_sizing.most_likely_parameters')
    def test_bet_size_reserve_default(self, mock_likely_parameters):
        """
        Tests for successful execution of 'bet_size_reserve' using default arguments, return_parameters=False.
        Function 'most_likely_parameters' needs to be patched because the 'M2N.mp_fit' method makes use of
        random numbers.
        """
        # Setup the test DataFrame.
        np.random.seed(0)
        sample_size = 500
        start_date = dt.datetime(2000, 1, 1)
        date_step = dt.timedelta(days=1)
        dates = np.array([start_date + i*date_step for i in range(sample_size)])
        shift_dt = np.array([dt.timedelta(days=d) for d in np.random.uniform(1., 20., sample_size)])
        dates_shifted = dates + shift_dt
        time_1 = pd.Series(data=dates_shifted, index=dates)
        df_events = time_1.to_frame()
        df_events = df_events.rename(columns={0: 't1'})
        df_events['p'] = np.random.uniform(0.0, 1.0, sample_size)
        df_events = df_events[['t1', 'p']]
        df_events['side'] = df_events['p'].apply(lambda x: 1 if x >= 0.5 else -1)
        # Calculate the correct results.
        events_active = get_concurrent_sides(df_events['t1'], df_events['side'])
        events_active['c_t'] = events_active['active_long'] - events_active['active_short']
        central_moments = [moment(events_active['c_t'].to_numpy(), moment=i) for i in range(1, 6)]
        raw_moments = raw_moment(central_moments=central_moments, dist_mean=events_active['c_t'].mean())
        m2n_test = M2N(raw_moments)
        test_results = m2n_test.mp_fit(epsilon=1e-5, factor=5, n_runs=25, variant=2, max_iter=10_000, num_workers=1)
        test_params = most_likely_parameters(test_results)
        mock_likely_parameters.return_value = test_params
        test_fit = [test_params[key] for key in ['mu_1', 'mu_2', 'sigma_1', 'sigma_2', 'p_1']]
        events_active['bet_size'] = events_active['c_t'].apply(lambda c: single_bet_size_mixed(c, test_fit))
        # Evaluate.
        df_bet = bet_size_reserve(df_events['t1'], df_events['side'], fit_runs=25)
        self.assertTrue(events_active.equals(df_bet))

    @patch('mlfinlab.bet_sizing.bet_sizing.most_likely_parameters')
    def test_bet_size_reserve_return_params(self, mock_likely_parameters):
        """
        Tests for successful execution of 'bet_size_reserve' using return_parameters=True.
        Function 'most_likely_parameters' needs to be patched because the 'M2N.mp_fit' method makes use of
        random numbers.
        """
        # Setup the test DataFrame.
        np.random.seed(0)
        sample_size = 500
        start_date = dt.datetime(2000, 1, 1)
        date_step = dt.timedelta(days=1)
        dates = np.array([start_date + i*date_step for i in range(sample_size)])
        shift_dt = np.array([dt.timedelta(days=d) for d in np.random.uniform(1., 20., sample_size)])
        dates_shifted = dates + shift_dt
        time_1 = pd.Series(data=dates_shifted, index=dates)
        df_events = time_1.to_frame()
        df_events = df_events.rename(columns={0: 't1'})
        df_events['p'] = np.random.uniform(0.0, 1.0, sample_size)
        df_events = df_events[['t1', 'p']]
        df_events['side'] = df_events['p'].apply(lambda x: 1 if x >= 0.5 else -1)
        # Calculate the correct results.
        events_active = get_concurrent_sides(df_events['t1'], df_events['side'])
        events_active['c_t'] = events_active['active_long'] - events_active['active_short']
        central_moments = [moment(events_active['c_t'].to_numpy(), moment=i) for i in range(1, 6)]
        raw_moments = raw_moment(central_moments=central_moments, dist_mean=events_active['c_t'].mean())
        m2n_test = M2N(raw_moments)
        test_results = m2n_test.mp_fit(epsilon=1e-5, factor=5, n_runs=25, variant=2, max_iter=10_000, num_workers=1)
        test_params = most_likely_parameters(test_results)
        mock_likely_parameters.return_value = test_params
        test_fit = [test_params[key] for key in ['mu_1', 'mu_2', 'sigma_1', 'sigma_2', 'p_1']]
        events_active['bet_size'] = events_active['c_t'].apply(lambda c: single_bet_size_mixed(c, test_fit))
        # Evaluate.
        eval_events, eval_params = bet_size_reserve(df_events['t1'], df_events['side'],
                                                    fit_runs=25, return_parameters=True)
        self.assertEqual(test_params, eval_params)
        self.assertTrue(events_active.equals(eval_events))
