"""
Contains unit demo tests
"""

import unittest
import mlfinlab.data_structures.code as code


class TestFunctions(unittest.TestCase):
    """
    Test demo functions
    """
    def test_addition(self):
        """
        Test addition
        """
        answer = 1 + 2
        self.assertTrue(answer == code.addition(1, 2))

    def test_subtraction(self):
        """
        Test subtraction
        """
        answer = 1 - 2
        self.assertTrue(answer == code.subtraction(1, 2))

    def test_fail(self):
        """
        Test a failure
        :return:
        """
        answer = 1 + 2
        self.assertFalse(answer == code.subtraction(1, 2))
