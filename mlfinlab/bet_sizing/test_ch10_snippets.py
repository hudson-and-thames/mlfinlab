"""
Tests the unit functions in ch10_snippets.py for calculating bet size.
"""

import unittest
import datetime as dt
import numpy as np
import pandas as pd

from scipy.stats import norm

from mlfinlab.bet_sizing.ch10_snippets import (get_signal, avg_active_signals,
                                               discrete_signal)


class TestCh10Snippets(unittest.TestCase):
    """
    Tests the following functions in ch10_snippets.py:
    1. get_signal
    2. avg_active_signals
    3. mp_avg_active_signals
    4. discrete_signal
    """

    def setUp(self):
        """
        Sets up the data to be used for the following tests.
        """
        # setup array values, length of 6
        prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775])
        side_arr = np.array([1, 1, -1, 1, -1, 1])
        dates = np.array([dt.datetime(2000, 1, 1) + i*dt.timedelta(days=1)
                          for i in range(6)])
        shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2]
        shift_dt = np.array([dt.timedelta(days=d) for d in shift_list])
        dates_shifted = dates + shift_dt

        # define inputs
        self.n_classes = 2
        self.step_size = 0.1

        # calculate bet sizes
        z = (prob_arr - 0.5) /  (prob_arr*(1-prob_arr))**0.5
        m = side_arr * (2 * norm.cdf(z) - 1)
        m_discrete = np.array([max(-1, min(1, m_i))
                               for m_i in np.round(m/0.1, 0)*0.1])

        # setup arrays as pd.Series
        self.prob = pd.Series(data=prob_arr, index=dates)
        self.side = pd.Series(data=side_arr, index=dates)
        self.t1 = pd.Series(data=dates_shifted, index=dates)
        self.bet_size = pd.Series(data=m, index=dates)
        self.bet_size_d = pd.Series(data=m_discrete, index=dates)


        # setup pd.Series as pd.DataFrames
        self.events = pd.concat(objs=[self.t1, self.prob, self.side], axis=1)
        self.events = self.events.rename(columns={0: 't1',
                                                  1: 'prob',
                                                  2: 'side'})
        self.events_2 = self.events.copy()
        self.events_2['signal'] = self.bet_size

        # calculation of the average active bets
        tp = set(self.events_2['t1'].to_numpy())
        tp = tp.union(self.events_2.index.to_numpy())
        tp = list(tp)
        tp.sort()
        avg_list = []
        for ti in tp:
            avg_list.append(self.events_2[(self.events_2.index <= ti)&\
                                   (self.events_2.t1 > ti)]['signal'].mean())
        self.avg_active = pd.Series(data=np.array(avg_list), index=tp).fillna(0)

    def test_get_signal(self):
        """
        Tests calculating the bet size from probability.
        """
        # test get_signal using a value for 'pred'
        test_bet_size_1 = get_signal(prob=self.prob, num_classes=self.n_classes,
                                     pred=self.side)
        self.assertEqual(self.bet_size.equals(test_bet_size_1), True)

        # test get_signal using no value for 'pred'
        test_bet_size_2 = get_signal(prob=self.prob, num_classes=self.n_classes)
        self.assertEqual(self.bet_size.abs().equals(test_bet_size_2), True)

    def test_avg_active_signals(self):
        """
        Tests the avg_active_signals function. Tests the molecular
        multiprocessing function mp_avg_active_signals implicitly.
        """
        test_avg_active = avg_active_signals(self.events_2)
        self.assertEqual(self.avg_active.equals(test_avg_active), True)

    def test_discrete_signal(self):
        """
        Tests the discrete_signal function.
        """
        test_bet_discrete = discrete_signal(self.bet_size, self.step_size)
        self.assertEqual(self.bet_size_d.equals(test_bet_discrete), True)

if __name__ == '__main__':
    unittest.main()
