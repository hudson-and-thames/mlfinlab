"""
Tests the unit functions in ch10_snippets.py for calculating bet size.
"""

import unittest
import datetime as dt
import numpy as np
import pandas as pd

from scipy.stats import norm

from mlfinlab.bet_sizing.ch10_snippets import get_signal, avg_active_signals, mp_avg_active_signals, discrete_signal


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
        dates = np.array([dt.datetime(2000, 1, 1) + i*dt.timedelta(days=1)
                          for i in range(6)])
        shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
        shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
        dates_shifted = dates + shift_dt

        # Calculate the test statistic and bet size.
        z_test = (prob_arr - 0.5) / (prob_arr*(1-prob_arr))**0.5
        m_signal = side_arr * (2 * norm.cdf(z_test) - 1)
        m_discrete = np.array([max(-1, min(1, m_i))
                               for m_i in np.round(m_signal/0.1, 0)*0.1])

        # Convert arrays to pd.Series for use as test arguments.
        self.prob = pd.Series(data=prob_arr, index=dates)
        self.side = pd.Series(data=side_arr, index=dates)
        self.t_1 = pd.Series(data=dates_shifted, index=dates)
        self.bet_size = pd.Series(data=m_signal, index=dates)
        self.bet_size_d = pd.Series(data=m_discrete, index=dates)

        # Convert pd.Series to pd.DataFrames for calculating correct results.
        self.events = pd.concat(objs=[self.t_1, self.prob, self.side], axis=1)
        self.events = self.events.rename(columns={0: 't1',
                                                  1: 'prob',
                                                  2: 'side'})
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
            avg_list.append(self.events_2[(self.events_2.index <= t_i)&\
                                   (self.events_2.t1 > t_i)]['signal'].mean())
        self.avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)

    def test_get_signal(self):
        """
        Tests calculating the bet size from probability.
        """
        # Test get_signal when supplying a value to argument 'pred'.
        test_bet_size_1 = get_signal(prob=self.prob, num_classes=2,
                                     pred=self.side)
        self.assertEqual(self.bet_size.equals(test_bet_size_1), True)

        # Test get_signal when no value provided for 'pred'.
        test_bet_size_2 = get_signal(prob=self.prob, num_classes=2)
        self.assertEqual(self.bet_size.abs().equals(test_bet_size_2), True)

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
        test_mp_avg_active = mp_avg_active_signals(self.events_2,
                                                   self.t_pnts)
        self.assertEqual(self.avg_active.equals(test_mp_avg_active), True)

    def test_discrete_signal(self):
        """
        Tests the discrete_signal function.
        """
        test_bet_discrete = discrete_signal(signal0=self.bet_size, step_size=0.1)
        self.assertEqual(self.bet_size_d.equals(test_bet_discrete), True)


if __name__ == '__main__':
    unittest.main()
