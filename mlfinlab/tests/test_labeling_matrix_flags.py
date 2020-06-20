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
        Set the file path for data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/close_df.csv'
        self.close = pd.read_csv(self.path, index_col='date', parse_dates=True)

    def test_init(self):
        """
        Tests that exceptions are raised correctly during initialization of the class if inputs are wrong.
        """
        # Length of data less than 10.
        close = self.close['spx']
        with self.assertRaises(Exception):
            MatrixFlagLabels(close[0:7], window=30)
        # Window less than 10.
        with self.assertRaises(Exception):
            MatrixFlagLabels(close[0:30], window=5)
        # Window greater than len(data).
        with self.assertRaises(Exception):
            MatrixFlagLabels(close[0:30], window=50)
        # Data is a DataFrame, not Series.
        with self.assertRaises(Exception):
            MatrixFlagLabels(self.close, window=60)

    def test_set_template(self):
        """
        Tests for user setting a new template. Also verifies that exception is raised for invalid template formats.
        """
        new_template = pd.DataFrame(np.random.randint(-3, 3, size=(10, 10)))
        Flags = MatrixFlagLabels(self.close['spx'], window=100)
        Flags.set_template(new_template)  # Correct template

        bad_shape = pd.DataFrame(np.random.randint(-3, 3, size=(9, 10)))  # Not 10 by 10
        with self.assertRaises(Exception):
            Flags.set_template(bad_shape)

        nan_template = new_template.copy()
        nan_template.iloc[2, 3] = np.nan  # Has NaN
        with self.assertRaises(Exception):
            Flags.set_template(nan_template)

    def test_transform_data(self):
        """
        Tests that the transform_data method gives the correct 10 by 10 matrix.
        """
        data = self.close['spx'][3:103]
        Flags = MatrixFlagLabels(data, window=30)
        test1 = Flags._transform_data(row_num=32, window=30)
        test2 = Flags._transform_data(row_num=100, window=90)

        test1_actual = pd.DataFrame([[0, 1. / 3, 0, 0, 1. / 3, 1. / 3, 0, 0, 0, 0],
                                     [0, 2. / 3, 0, 0, 1. / 3, 0, 0, 0, 0, 0],
                                     [0, 0, 1. / 3, 2. / 3, 0, 0, 0, 0, 0, 0],
                                     [1. / 3, 0, 0, 0, 0, 1. / 3, 0, 0, 0, 0],
                                     [1. / 3, 0, 1. / 3, 1. / 3, 1. / 3, 0, 0, 0, 0, 0],
                                     [1. / 3, 0, 1. / 3, 0, 0, 0, 0, 0, 1. / 3, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1. / 3, 1. / 3, 1. / 3],
                                     [0, 0, 0, 0, 0, 0, 0, 1. / 3, 1. / 3, 1. / 3],
                                     [0, 0, 0, 0, 0, 0, 2. / 3, 0, 0, 1. / 3],
                                     [0, 0, 0, 0, 0, 1. / 3, 1. / 3, 1. / 3, 0, 0]])
        test2_column_actual = pd.Series([0, 0, 0, 0, 0, 0, 0, 1. / 3, 1. / 9, 5. / 9])
        pd.testing.assert_frame_equal(test1, test1_actual)
        pd.testing.assert_series_equal(test2[1], test2_column_actual, check_names=False)  # Check 1 column

        # Assert that all columns sum to 1.
        self.assertTrue(test2.sum(axis=0).values.all() == np.array([1.] * 10).all())

    def test_apply_template(self):
        """
        Tests for the apply template to matrix. A matrix is used which satisfies the constraints of transform_data.
        Then, the template is changed, and the applied to the same matrix.
        """
        matrix = pd.DataFrame([[0, 1. / 3, 0, 0, 1. / 3, 1. / 3, 0, 0, 0, 0],
                               [0, 2. / 3, 0, 0, 1. / 3, 0, 0, 0, 0, 0],
                               [0, 0, 1. / 3, 2. / 3, 0, 0, 0, 0, 0, 0],
                               [1. / 3, 0, 0, 0, 0, 1. / 3, 0, 0, 0, 0],
                               [1. / 3, 0, 1. / 3, 1. / 3, 1. / 3, 0, 0, 0, 0, 0],
                               [1. / 3, 0, 1. / 3, 0, 0, 0, 0, 0, 1. / 3, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1. / 3, 1. / 3, 1. / 3],
                               [0, 0, 0, 0, 0, 0, 0, 1. / 3, 1. / 3, 1. / 3],
                               [0, 0, 0, 0, 0, 0, 2. / 3, 0, 0, 1. / 3],
                               [0, 0, 0, 0, 0, 1. / 3, 1. / 3, 1. / 3, 0, 0]])

        Flags = MatrixFlagLabels(self.close['spx'], window=100)  # Inputs don't matter.
        test3 = Flags._apply_template_to_matrix(matrix, Flags.template)
        self.assertAlmostEqual(test3, 2.6666666667)

        # Replace the template with another bull template proposed in Cervello-Royo's paper.
        new_template = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                                     [0, 0, 0, -1, -2, -2, -2, -2, -2, -2],
                                     [0, 0, -1, -3, -3, -3, -3, -3, -3, -3],
                                     [0, -1, -3, -5, -5, -5, -5, -5, -5, -5],
                                     [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
                                     [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
                                     [5, -1, -5, -5, -5, -5, -5, -5, -5, -5]])

        Flags.set_template(new_template)
        test4 = Flags._apply_template_to_matrix(matrix, Flags.template)
        self.assertAlmostEqual(test4, -22.666666667)

    def test_apply_labeling(self):
        """
        Test for the function the users would actually use, for creating full labels from matrix.
        """
        data = self.close['spx'][0:100]
        Flags = MatrixFlagLabels(data=data, window=60)
        test5 = Flags.apply_labeling_matrix()

        # Verify that the output has 40 (100 - 60) rows.
        self.assertTrue(len(test5) == 40)

        # Check values.
        test5_subset = test5[10:20]
        test5_subset_actual = pd.Series([-0.91666667, -1.25, -1.75, -2.16666667, -2.66666667, -2.66666667, -2.75,
                                         -2.5, -2.58333333, -2.66666667], index=test5_subset.index)
        pd.testing.assert_series_equal(test5_subset, test5_subset_actual)
