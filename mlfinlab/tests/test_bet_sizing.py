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

    def test_bet_size_budget_div_zero(self):
        """
        Tests for successful handling of events DataFrames that result in a maximum number of
        concurrent bet sides of zero.
        """
        # Setup the test DataFrame.
        dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
        shift_dt = np.array([dt.timedelta(days=0.5*i+1) for i in range(5)])
        dates_shifted_test = dates_test + shift_dt
        events_test = pd.DataFrame(data=[[0.55, 1], [0.7, 1], [0.95, 1], [0.65, 1], [0.85, 1]],
                                   columns=['prob', 'side'],
                                   index=dates_test)
        events_test['t1'] = dates_shifted_test
        # Calculate correct results.
        events_result = get_concurrent_sides(events_test['t1'], events_test['side'])
        max_active_long, max_active_short = events_result['active_long'].max(), events_result['active_short'].max()
        avg_long = events_result['active_long'] / max_active_long if max_active_long > 0 else 0
        avg_short = events_result['active_short'] / max_active_short if max_active_short > 0 else 0
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
        m2n_test = M2N(raw_moments, epsilon=1e-5, factor=5, n_runs=25, variant=2, max_iter=10_000, num_workers=1)
        test_results = m2n_test.mp_fit()
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
        m2n_test = M2N(raw_moments, epsilon=1e-5, factor=5, n_runs=25, variant=2, max_iter=10_000, num_workers=1)
        test_results = m2n_test.mp_fit()
        test_params = most_likely_parameters(test_results)
        mock_likely_parameters.return_value = test_params
        test_fit = [test_params[key] for key in ['mu_1', 'mu_2', 'sigma_1', 'sigma_2', 'p_1']]
        events_active['bet_size'] = events_active['c_t'].apply(lambda c: single_bet_size_mixed(c, test_fit))
        # Evaluate.
        eval_events, eval_params = bet_size_reserve(df_events['t1'], df_events['side'],
                                                    fit_runs=25, return_parameters=True)
        self.assertEqual(test_params, eval_params)
        self.assertTrue(events_active.equals(eval_events))


class TestConfirmAndCastToDf(unittest.TestCase):
    """
    Tests the 'confirm_and_cast_to_df' function.
    """
    def test_cast_to_df_all_series(self):
        """
        Tests for successful execution of 'confirm_and_cast_to_df' when all dictionary values are pandas.Series.
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
        d_events = {col: events_test[col] for col in list(events_test.columns)}
        # Evaluate.
        df_results = confirm_and_cast_to_df(d_events)
        self.assertTrue(events_test.equals(df_results))

    def test_cast_to_df_one_series(self):
        """
        Tests for successful execution of 'confirm_and_cast_to_df' when only one of the dictionary values are
        a pandas.Series.
        """
        # Setup the test DataFrame.
        dates_test = np.array([dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)])
        max_pos, market_price, forecast_price = 55, 75.00, 80.00
        events_test = pd.DataFrame(data=[[25, max_pos, market_price, forecast_price],
                                         [35, max_pos, market_price, forecast_price],
                                         [45, max_pos, market_price, forecast_price],
                                         [40, max_pos, market_price, forecast_price],
                                         [30, max_pos, market_price, forecast_price]],
                                   columns=['pos', 'max_pos', 'm_p', 'f'],
                                   index=dates_test)
        d_events = {'pos': events_test['pos'], 'max_pos': max_pos, 'm_p': market_price, 'f': forecast_price}
        # Evaluate.
        df_results = confirm_and_cast_to_df(d_events)
        self.assertTrue(np.allclose(events_test.to_numpy(), df_results.to_numpy(), 1e-9))

    def test_cast_to_df_no_series(self):
        """
        Tests for successful execution of 'confirm_and_cast_to_df' when none of the dictionary values are
        a pandas.Series.
        """
        # Setup the test DataFrame.
        pos, max_pos, market_price, forecast_price = 35, 55, 75.00, 80.00
        events_test = pd.DataFrame(data=[[pos, max_pos, market_price, forecast_price]],
                                   columns=['pos', 'max_pos', 'm_p', 'f'])
        d_events = {'pos': pos, 'max_pos': max_pos, 'm_p': market_price, 'f': forecast_price}
        # Evaluate.
        df_results = confirm_and_cast_to_df(d_events)
        self.assertTrue(np.allclose(events_test.to_numpy(), df_results.to_numpy(), 1e-9))


class TestGetConcurrentSides(unittest.TestCase):
    """
    Tests the function 'get_concurrent_sides' for successful operation.
    """
    def test_get_concurrent_sides_default(self):
        """
        Tests for the successful execution of 'get_concurrent_sides'. Since there are no options or branches,
        there are no additional test cases beyond default.
        """
        # Setup the test DataFrame.
        np.random.seed(0)
        sample_size = 100
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
        # Calculate correct result.
        events_test = df_events.copy()
        events_test['active_long'] = 0
        events_test['active_short'] = 0
        for idx in events_test.index:
            df_long_active_idx = set(events_test[(events_test.index <= idx) & (events_test['t1'] > idx) \
                                        & (events_test['side'] > 0)].index)
            events_test.loc[idx, 'active_long'] = len(df_long_active_idx)
            df_short_active_idx = set(events_test[(events_test.index <= idx) & (events_test['t1'] > idx) \
                                        & (events_test['side'] < 0)].index)
            events_test.loc[idx, 'active_short'] = len(df_short_active_idx)
        events_test = events_test[['active_long', 'active_short']]
        # Evaluate.
        df_results = get_concurrent_sides(df_events['t1'], df_events['side'])
        self.assertTrue(np.allclose(events_test.to_numpy(), df_results[['active_long', 'active_short']].to_numpy(), 1e-9))


class TestCdfMixture(unittest.TestCase):
    """
    Tests the 'cdf_mixture' function.
    """
    def test_cdf_mixture_default(self):
        """
        Tests for the successful execution of the 'cdf_mixture' function. Since there are no options or branches,
        there are no additional test cases beyond default.
        """
        # Setup the test data.
        x_value = 5.0
        mu_1, mu_2, sigma_1, sigma_2, p_1 = -1.0, 4.0, 2.0, 1.5, 0.4
        params = [mu_1, mu_2, sigma_1, sigma_2, p_1]
        # Calculate the expected results.
        expected_result = (p_1 * norm.cdf(x_value, mu_1, sigma_1)) + ((1-p_1) * norm.cdf(x_value, mu_2, sigma_2))
        # Evaluate.
        self.assertEqual(expected_result, cdf_mixture(x_value, params))


class TestSingleBetSizeMixed(unittest.TestCase):
    """
    Tests the 'single_bet_size_mixed' function.
    """
    def test_single_bet_size_mixed_above_zero(self):
        """
        Tests for the successful execution of the 'single_bet_size_mixed' function where the 'c_t' parameter is
        greater than zero.
        """
        # Setup the test data.
        c_t = 5.0
        mu_1, mu_2, sigma_1, sigma_2, p_1 = -1.0, 4.0, 2.0, 1.5, 0.4
        params = [mu_1, mu_2, sigma_1, sigma_2, p_1]
        # Calculate the expected result.
        expected_result = (cdf_mixture(c_t, params) - cdf_mixture(0, params)) / (1 - cdf_mixture(0, params))
        # Evaluate.
        self.assertEqual(expected_result, single_bet_size_mixed(c_t, params))

    def test_single_bet_size_mixed_below_zero(self):
        """
        Tests for the successful execution of the 'single_bet_size_mixed' function where the 'c_t' parameter is
        less than zero.
        """
        # Setup the test data.
        c_t = -4.0
        mu_1, mu_2, sigma_1, sigma_2, p_1 = -1.0, 4.0, 2.0, 1.5, 0.4
        params = [mu_1, mu_2, sigma_1, sigma_2, p_1]
        # Calculate the expected result.
        expected_result = (cdf_mixture(c_t, params) - cdf_mixture(0, params)) / cdf_mixture(0, params)
        # Evaluate.
        self.assertEqual(expected_result, single_bet_size_mixed(c_t, params))
