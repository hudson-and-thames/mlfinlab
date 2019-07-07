"""
Tests the unit functions in ch10_snippets.py for calculating bet size.
"""

import unittest
import datetime as dt
import numpy as np
import pandas as pd

from scipy.stats import norm

from mlfinlab.bet_sizing.ch10_snippets import (bet_size_sigmoid,
                                               get_t_pos_sigmoid,
                                               inv_price_sigmoid,
                                               limit_price_sigmoid,
                                               get_w_sigmoid)


class TestCh10Snippets(unittest.TestCase):
    """
    Tests the following functions in ch10_snippets.py:
    - bet_size_sigmoid
    - get_t_pos_sigmoid
    - inv_price_sigmoid
    - limit_price_sigmoid
    - get_w_sigmoid
    """

    def setUp(self):
        """
        Sets up the data to be used for the following tests.
        """
        

    def test_bet_size_sigmoid(self):
        """
        Tests calculating the bet size dynamically using a sigmoid function.
        """
        # test get_signal using a value for 'pred'
        test_bet_size_1 = get_signal(prob=self.prob, num_classes=self.n_classes,
                                     pred=self.side)
        self.assertEqual(self.bet_size.equals(test_bet_size_1), True)

        # test get_signal using no value for 'pred'
        test_bet_size_2 = get_signal(prob=self.prob, num_classes=self.n_classes)
        self.assertEqual(self.bet_size.abs().equals(test_bet_size_2), True)

    def test_get_t_pos_sigmoid(self):
        """
        Tests calculating the bet size dynamically using a sigmoid function.
        """
    
    def test_inv_price_sigmoid(self):
        """
        Tests calculating the bet size dynamically using a sigmoid function.
        """
    
    def test_limit_price_sigmoid(self):
        """
        Tests calculating the bet size dynamically using a sigmoid function.
        """
    
    def test_get_w_sigmoid(self):
        """
        Tests calculating the bet size dynamically using a sigmoid function.
        """


if __name__ == '__main__':
    unittest.main()
