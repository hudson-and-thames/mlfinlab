import unittest
import mlfinlab.data_structures.code as code


class TestFunctions(unittest.TestCase):

    def test_addition(self):
        answer = 1 + 2
        self.assertTrue(answer == code.addition(1, 2))

    def test_subtraction(self):

        answer = 1 - 2
        self.assertTrue(answer == code.subtraction(1, 2))

    def test_fail(self):

        answer = 1 + 2
        self.assertFalse(answer == code.subtraction(1, 2))
