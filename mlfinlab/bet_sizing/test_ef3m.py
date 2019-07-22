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
        m_1 = p_1*u_1 + p_2*u_2
        m_2 = p_1*(s_1**2 + u_1**2) + p_2*(s_2**2 + u_2**2)
        m_3 = p_1*(3*s_1**2*u_1 + u_1**3) + p_2*(3*s_2**2*u_2 + u_2**3)
        m_4 = p_1*(3*s_1**4 + 6*s_1**2*u_1**2 + u_1**4) + p_2*(3*s_2**4 + 6*s_2**2*u_2**2 + u_2**4)
        m_5 = p_1*(15*s_1**4*u_1 + 10*s_1**2*u_1**3 + u_1**5) + p_2*(15*s_2**4*u_2 + 10*s_2**2*u_2**3 + u_2**5)
        test_params = [u_1, u_2, s_1, s_2, p_1]
        test_mmnts = [m_1, m_2, m_3, m_4, m_5]
        # Create M2N object.
        m2n_test = M2N(test_mmnts)
        # Check self-return method.
        m2n_test.get_moments(test_params, return_result=False)
        self.assertEqual(test_mmnts, m2n_test.new_moments)
        # Check the function when 'return_value' is True.
        result_mmnts = m2n_test.get_moments(test_params, return_result=True)
        self.assertEqual(test_mmnts, result_mmnts)

    def test_iter_4(self):
        """
        Tests the 'iter_4' method in the M2N class.
        """
        moments_test = [1, 2, 3, 4, 5]
        m2n_test = M2N(moments_test)
        # Test for 'Validity check 1'.
        param_results = m2n_test.iter_4(3, 1)
        self.assertTrue(not param_results)
