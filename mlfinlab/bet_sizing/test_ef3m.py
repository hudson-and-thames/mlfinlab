"""
Tests the unit functions in ef3m.py for calculating fitting a mixture of 2 Gaussian distributions.
"""

import unittest
import numpy as np
import pandas as pd



from mlfinlab.bet_sizing.ef3m import M2N


class TestM2N(unittest.TestCase):
    """
    Tests the methods of the M2N class.
    """

    def test_m2n_constructor(self):
        """
        Tests that the constructor of the M2N class executes properly.
        """
        moments_test = [1, 2, 3, 4, 5]
        m2n_test = M2N(moments_test)
        # Confirm that the initial attributes get set properly.
        self.assertEqual(m2n_test.moments, moments_test)
        self.assertEqual(m2n_test.new_moments, [0, 0, 0, 0, 0])
        self.assertEqual(m2n_test.parameters, [0, 0, 0, 0, 0])
        self.assertEqual(m2n_test.error, sum([moments_test[i]**2 for i in range(len(moments_test))]))

    def test_get_moments(self):
        """
        Tests the 'get_moments' method of the M2N class.
        """
        u_1, u_2, s_1, s_2, p_1 = [2.1, 4.3, 1.1, 0.7, 0.3]
        p_2 = 1 - p_1
        m_1 = p_1
