"""
Tests Buy and Hold Strategy(BAH).
"""

import unittest
import os
import numpy as np
import pandas as pd


class TestBAH(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the BAH class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")


if __name__ == '__main__':
    unittest.main()
