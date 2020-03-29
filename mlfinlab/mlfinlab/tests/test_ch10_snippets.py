"""
Tests the unit functions in ch10_snippets.py for calculating bet size.
"""

import unittest
import warnings
import datetime as dt
import numpy as np
import pandas as pd

from scipy.stats import norm

from mlfinlab.bet_sizing.ch10_snippets import get_signal, avg_active_signals, mp_avg_active_signals, discrete_signal
from mlfinlab.bet_sizing.ch10_snippets import (bet_size_sigmoid, bet_size_power, bet_size, get_target_pos_sigmoid, get_target_pos_power, get_target_pos,
                                               inv_price_sigmoid, inv_price_power, inv_price, limit_price_sigmoid, limit_price_power, limit_price,
                                               get_w_sigmoid, get_w_power, get_w)


class TestCh10Snippets(unittest.TestCase):
    """
    Tests the following functions in ch10_snippets.py:
    - get_signal
    - avg_active_signals
    - mp_avg_active_signals
    - discrete_signal
    """

    def setUp(self):
        """
        Sets up the data to be used for the following tests.
        """
        # ======================================================================
        # Unit test setup for get_signal, avg_active_signals,
        # mp_avg_active_signals, and discrete_signal.

        # Setup the array of values, length of 6 for testing.
        prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
        side_arr = np.array([1, 1, -1, 1, -1, 1])
        dates = np.array([dt.datetime(2000, 1, 1) + i*dt.timedelta(days=1) for i in range(6)])

        shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
        shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
        dates_shifted = dates + shift_dt

        # Calculate the test statistic and bet size.
        z_test = (prob_arr - 0.5) / (prob_arr*(1-prob_arr))**0.5
        m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
        m_discrete = np.array([max(-1, min(1, m_i)) for m_i in np.round(m_signal/0.1, 0)*0.1])

        # Convert arrays to pd.Series for use as test arguments.
        self.prob = pd.Series(data=prob_arr, index=dates)
        self.side = pd.Series(data=side_arr, index=dates)
        self.t_1 = pd.Series(data=dates_shifted, index=dates)
        self.bet_size = pd.Series(data=m_signal, index=dates)
        self.bet_size_d = pd.Series(data=m_discrete, index=dates)

        # Convert pd.Series to pd.DataFrames for calculating correct results.
        self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
        self.events = self.events.rename(columns={0: 't1', 1: 'prob', 2: 'side'})
        self.events_2 = self.events.copy()
        self.events_2['signal'] = self.bet_size

        # Calculation of the average active bets.
        t_p = set(self.events_2['t1'].to_numpy())
        t_p = t_p.union(self.events_2.index.to_numpy())
        t_p = list(t_p)
        t_p.sort()
        self.t_pnts = t_p.copy()
        avg_list = []
        for t_i in t_p:
            avg_list.append(self.events_2[(self.events_2.index <= t_i) & (self.events_2.t1 > t_i)]['signal'].mean())
        self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

    def test_get_signal(self):
        """
        Tests calculating the bet size from probability.
        """
        # Test get_signal when supplying a value to argument 'pred'.
        test_bet_size_1 = get_signal(prob=self.prob, num_classes=2, pred=self.side)
        self.assertEqual(self.bet_size.equals(test_bet_size_1), True)

        # Test get_signal when no value provided for 'pred'.
        test_bet_size_2 = get_signal(prob=self.prob, num_classes=2)
        self.assertEqual(self.bet_size.abs().equals(test_bet_size_2), True)

        # Test for prob.shape[0] == 0.
        df_empty = pd.DataFrame({'a': []})
        return_val = get_signal(df_empty, 2)
        self.assertIsInstance(return_val, pd.Series)
        self.assertEqual(0, len(return_val))

    def test_avg_active_signals(self):
        """
        Tests the avg_active_signals function. Also implicitly tests the
        molecular multiprocessing function mp_avg_active_signals.
        """
        test_avg_active = avg_active_signals(self.events_2)
        self.assertEqual(self.avg_active.equals(test_avg_active), True)

    def test_mp_avg_active_signals(self):
        """
        An explicit test of the mp_avg_active_signals subroutine.
        """
        test_mp_avg_active = mp_avg_active_signals(self.events_2, self.t_pnts)
        self.assertEqual(self.avg_active.equals(test_mp_avg_active), True)

    def test_discrete_signal(self):
        """
        Tests the discrete_signal function.
        """
        test_bet_discrete = discrete_signal(signal0=self.bet_size, step_size=0.1)
        self.assertEqual(self.bet_size_d.equals(test_bet_discrete), True)


# =====================================================================================================================
# Test cases for functions used in calculating dynamic bet size.
class TestBetSize(unittest.TestCase):
    """
    Test case for bet_size, bet_size_sigmoid, and bet_size_power.
    """
    def test_bet_size_sigmoid(self):
        """
        Tests successful execution of 'bet_size_sigmoid'.
        """
        x_div, w_param = 15, 7.5
        m_test = x_div / np.sqrt(w_param + x_div*x_div)
        self.assertAlmostEqual(m_test, bet_size_sigmoid(w_param, x_div), 7)

    def test_bet_size_power(self):
        """
        Tests successful execution of 'bet_size_power'.
        """
        x_div, w_param = 0.4, 1.5
        m_test = np.sign(x_div) * abs(x_div)**w_param
        self.assertAlmostEqual(m_test, bet_size_power(w_param, x_div), 7)

    def test_bet_size_power_value_error(self):
        """
        Tests successful raising of ValueError in 'bet_size_power'.
        """
        self.assertRaises(ValueError, bet_size_power, 2, 1.5)

    def test_bet_size_power_return_zero(self):
        """
        Tests that the function returns zero if the price divergence provided is zero.
        """
        self.assertEqual(0.0, bet_size_power(2, 0.0))

    def test_bet_size(self):
        """
        Test excution in all function modes of 'bet_size'.
        """
        x_div_sig, w_param_sig = 25, 3.5
        m_test_sig = x_div_sig / np.sqrt(w_param_sig + x_div_sig*x_div_sig)
        self.assertAlmostEqual(m_test_sig, bet_size(w_param_sig, x_div_sig, 'sigmoid'), 7)
        x_div_pow, w_param_pow = 0.7, 2.1
        m_test_pow = np.sign(x_div_pow) * abs(x_div_pow)**w_param_pow
        self.assertAlmostEqual(m_test_pow, bet_size(w_param_pow, x_div_pow, 'power'), 7)

    def test_bet_size_key_error(self):
        """
        Tests for the KeyError in the event that an invalid function is provided to 'func'.
        """
        self.assertRaises(KeyError, bet_size, 2, 3, 'NotAFunction')


class TestGetTPos(unittest.TestCase):
    """
    Test case for get_target_pos, get_target_pos_sigmoid, and get_target_pos_power.
    """
    def test_get_target_pos_sigmoid(self):
        """
        Tests successful execution of 'get_target_pos_sigmoid'.
        """
        f_i, m_p = 34.6, 21.9
        x_div = f_i - m_p
        w_param = 2.5
        max_pos = 200
        pos_test = int(max_pos*(x_div / np.sqrt(w_param + x_div*x_div)))
        self.assertAlmostEqual(pos_test, get_target_pos_sigmoid(w_param, f_i, m_p, max_pos), 7)

    def test_get_target_pos_power(self):
        """
        Tests successful execution of 'get_target_pos_power'.
        """
        f_i, m_p = 34.6, 34.1
        x_div = f_i - m_p
        w_param = 2.1
        max_pos = 100
        pos_test = int(max_pos*(np.sign(x_div) * abs(x_div)**w_param))
        self.assertAlmostEqual(pos_test, get_target_pos_power(w_param, f_i, m_p, max_pos), 7)

    def test_get_target_pos(self):
        """
        Tests successful execution in 'sigmoid' and 'power' function variants of 'get_target_pos'.
        """
        f_i_sig, m_p_sig = 31.6, 22.9
        x_div_sig = f_i_sig - m_p_sig
        w_param_sig = 2.6
        max_pos_sig = 220
        pos_test_sig = int(max_pos_sig*(x_div_sig / np.sqrt(w_param_sig + x_div_sig*x_div_sig)))
        self.assertAlmostEqual(pos_test_sig, get_target_pos(w_param_sig, f_i_sig, m_p_sig, max_pos_sig, 'sigmoid'), 7)
        f_i_pow, m_p_pow = 34.8, 34.1
        x_div_pow = f_i_pow - m_p_pow
        w_param_pow = 2.9
        max_pos_pow = 175
        pos_test_pow = int(max_pos_pow*(np.sign(x_div_pow) * abs(x_div_pow)**w_param_pow))
        self.assertAlmostEqual(pos_test_pow, get_target_pos(w_param_pow, f_i_pow, m_p_pow, max_pos_pow, 'power'), 7)

    def test_get_target_pos_key_error(self):
        """
        Tests for the KeyError in 'get_target_pos' in the case that an invalid value for 'func' is passed.
        """
        self.assertRaises(KeyError, get_target_pos, 1, 2, 1, 5, 'NotAFunction')


class TestInvPrice(unittest.TestCase):
    """
    Tests for functions 'inv_price', 'inv_price_sigmoid', and 'inv_price_power'.
    """
    def test_inv_price_sigmoid(self):
        """
        Test for the successful execution of 'inv_price_sigmoid'.
        """
        f_i_sig, w_sig, m_sig = 35.19, 9.32, 0.72
        inv_p_sig = f_i_sig - m_sig * np.sqrt(w_sig/(1-m_sig*m_sig))
        self.assertAlmostEqual(inv_p_sig, inv_price_sigmoid(f_i_sig, w_sig, m_sig), 7)

    def test_inv_price_power(self):
        """
        Test for the successful execution of 'inv_price_sigmoid'.
        """
        f_i_pow, w_pow, m_pow = 35.19, 3.32, 0.72
        inv_p_pow = f_i_pow - np.sign(m_pow) * abs(m_pow)**(1/w_pow)
        self.assertAlmostEqual(inv_p_pow, inv_price_power(f_i_pow, w_pow, m_pow), 7)
        self.assertEqual(f_i_pow, inv_price_power(f_i_pow, w_pow, 0.0))

    def test_inv_price(self):
        """
        Test for successful execution of 'inv_price' function under different function options.
        """
        f_i_sig, w_sig, m_sig = 87.19, 7.34, 0.82
        inv_p_sig = f_i_sig - m_sig * np.sqrt(w_sig/(1-m_sig*m_sig))
        self.assertAlmostEqual(inv_p_sig, inv_price(f_i_sig, w_sig, m_sig, 'sigmoid'), 7)
        f_i_pow, w_pow, m_pow = 129.19, 4.02, 0.81
        inv_p_pow = f_i_pow - np.sign(m_pow) * abs(m_pow)**(1/w_pow)
        self.assertAlmostEqual(inv_p_pow, inv_price(f_i_pow, w_pow, m_pow, 'power'), 7)

    def test_inv_price_key_error(self):
        """
        Test for successful raising of KeyError in response to invalid choice of 'func' argument.
        """
        self.assertRaises(KeyError, inv_price, 12, 1.5, 0.7, 'NotAFunction')


class TestLimitPrice(unittest.TestCase):
    """
    Tests the functions 'limit_price_sigmoid', 'limit_price_power', and 'limit_price'.
    """
    def test_limit_price_sigmoid(self):
        """
        Test successful execution of 'limit_price_sigmoid' function.
        """
        t_pos_sig, pos_sig, f_sig, w_sig, max_pos_sig = 124, 112, 165.50, 8.44, 150
        sum_inv_price_sig = sum([inv_price_sigmoid(f_sig, w_sig, j/float(max_pos_sig))
                                 for j in range(abs(pos_sig+np.sign(t_pos_sig-pos_sig)), abs(t_pos_sig+1))])
        limit_p_sig = (1/abs(t_pos_sig-pos_sig)) * sum_inv_price_sig
        self.assertAlmostEqual(limit_p_sig, limit_price_sigmoid(t_pos_sig, pos_sig, f_sig, w_sig, max_pos_sig), 7)

    def test_limit_price_sigmoid_return_nan(self):
        """
        Tests for the successful return of np.nan in the case that the target position is the same as the
        current position.
        """
        self.assertTrue(np.isnan(limit_price_sigmoid(1, 1, 123, 21, 234)))

    def test_limit_price_power(self):
        """
        Test successful execution of 'limit_price_power' function.
        """
        t_pos_pow, pos_pow, f_pow, w_pow, max_pos_pow = 101, 95, 195.70, 3.44, 130
        sum_inv_price_pow = sum([inv_price_power(f_pow, w_pow, j/float(max_pos_pow))
                                 for j in range(abs(pos_pow+np.sign(t_pos_pow-pos_pow)), abs(t_pos_pow+1))])
        limit_p_pow = (1/abs(t_pos_pow-pos_pow)) * sum_inv_price_pow
        self.assertAlmostEqual(limit_p_pow, limit_price_power(t_pos_pow, pos_pow, f_pow, w_pow, max_pos_pow), 7)

    def test_limit_price_key_error(self):
        """
        Tests raising of the KeyError due to invalid choice of 'func' argument.
        """
        self.assertRaises(KeyError, limit_price, 231, 221, 110, 3.4, 250, 'NotAFunction')


class TestGetW(unittest.TestCase):
    """
    Tests the functions 'get_w_sigmoid', 'get_w_power', and 'get_w'.
    """
    def test_get_w_sigmoid(self):
        """
        Tests successful execution of 'get_w_sigmoid' function.
        """
        x_sig, m_sig = 24.2, 0.98
        w_sig = x_sig**2 * (m_sig**-2 - 1)
        self.assertAlmostEqual(w_sig, get_w_sigmoid(x_sig, m_sig), 7)

    def test_get_w_power(self):
        """
        Tests successful execution of 'get_w_power' function.
        """
        x_pow, m_pow = 0.9, 0.76
        w_pow = np.log(m_pow/np.sign(x_pow)) / np.log(abs(x_pow))
        self.assertAlmostEqual(w_pow, get_w_power(x_pow, m_pow), 7)

    def test_get_w_power_value_error(self):
        """
        Tests that a ValueError is raised if the price divergence 'x' is not between -1 and 1, inclusive.
        """
        self.assertRaises(ValueError, get_w_power, 1.2, 0.8)

    def test_get_w_power_warning(self):
        """
        Tests that a UserWarning is raised if 'w' is calcualted to be less than zero, and returns a zero.
        """
        with warnings.catch_warnings(record=True) as warn_catch:
            warnings.simplefilter("always")
            w_param = get_w_power(0.1, 2)
            print(warn_catch[0].category)
            self.assertTrue(len(warn_catch) == 1)
            self.assertEqual(0, w_param)
            self.assertTrue('User' in str(warn_catch[0].category))

    def test_get_w_key_error(self):
        """
        Tests that a KeyError is raised if an invalid function is passed to argument 'func'.
        """
        self.assertRaises(KeyError, get_w, 0.6, 0.9, 'NotAFunction')
