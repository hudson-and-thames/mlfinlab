# pylint: disable=missing-module-docstring

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.matrix_flags import MatrixFlagLabels


class TestMatrixFlagLabels(unittest.TestCase):
    """
    Tests for the matrix flags labeling method.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(self.path, index_col='Date')
        self.idx10 = self.data[:10].index

    def test_set_template(self):
        """
        Tests for user setting a new template. Also verifies that exception is raised for invalid template formats.
        """
        self.assertEqual(True, False)



