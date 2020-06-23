# pylint: disable=missing-module-docstring
# pylint: disable=protected-access

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
        flags = MatrixFlagLabels(self.close['spx'], window=100)
        flags.set_template(new_template)  # Correct template

        bad_shape = pd.DataFrame(np.random.randint(-3, 3, size=(9, 10)))  # Not 10 by 10
        with self.assertRaises(Exception):
            flags.set_template(bad_shape)

        nan_template = new_template.copy()
        nan_template.iloc[2, 3] = np.nan  # Has NaN
        with self.assertRaises(Exception):
            flags.set_template(nan_template)

    def test_transform_data(self):
        """
        Tests that the transform_data method gives the correct 10 by 10 matrix.
        """
        data = self.close['spx'][3:103]
        flags = MatrixFlagLabels(data, window=30)
        test1 = flags._transform_data(row_num=32, window=30)
        test2 = flags._transform_data(row_num=100, window=90)

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

        flags = MatrixFlagLabels(self.close['spx'], window=100, template_name='leigh_bull')
        test3 = flags._apply_template_to_matrix(matrix, flags.template)
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

        flags.set_template(new_template)
        test4 = flags._apply_template_to_matrix(matrix, flags.template)
        self.assertAlmostEqual(test4, -22.666666667)

    def test_apply_labeling(self):
        """
        Test for the function the users would actually use, for creating full labels from matrix.
        """
        data = self.close['spx'][0:100]
        flags = MatrixFlagLabels(prices=data, window=60, template_name='leigh_bull')
        test5 = flags.apply_labeling_matrix()

        # Verify that the output has 40 (100 - 60) rows.
        self.assertTrue(len(test5) == 40)

        # Check values.
        test5_subset = test5[10:20]
        test5_subset_actual = pd.Series([-0.91666667, -1.25, -1.75, -2.16666667, -2.66666667, -2.66666667, -2.75,
                                         -2.5, -2.58333333, -2.66666667], index=test5_subset.index)
        pd.testing.assert_series_equal(test5_subset, test5_subset_actual)

    def test_threshold(self):
        """
        Tests for when threshold is desired.
        """
        data = self.close['spx'][425:500]
        idx = data[60:].index
        flags = MatrixFlagLabels(prices=data, window=60, template_name='leigh_bull')
        test5 = flags.apply_labeling_matrix(threshold=4)
        test6 = flags.apply_labeling_matrix(threshold=-1.5)
        test5_actual = pd.Series([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], index=idx)
        test6_actual = pd.Series([1]*15, index=idx)
        pd.testing.assert_series_equal(test5, test5_actual)
        pd.testing.assert_series_equal(test6, test6_actual)

    def test_template_init(self):
        """
        Checks that other templates are given correctly.
        """
        close = self.close['spx']
        leigh_bull = pd.DataFrame([[.5, 0, -1, -1, -1, -1, -1, -1, -1, 0],
                                   [1, 0.5, 0, -0.5, -1, -1, -1, -1, -0.5, 0],
                                   [1, 1, 0.5, 0, -0.5, -0.5, -0.5, -0.5, 0, 0.5],
                                   [0.5, 1, 1, 0.5, 0, -0.5, -0.5, -0.5, 0, 1],
                                   [0, 0.5, 1, 1, 0.5, 0, 0, 0, 0.5, 1],
                                   [0, 0, 0.5, 1, 1, 0.5, 0, 0, 1, 1],
                                   [-0.5, 0, 0, 0.5, 1, 1, 0.5, 0.5, 1, 1],
                                   [-0.5, -1, 0, 0, 0.5, 1, 1, 1, 1, 0],
                                   [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 0, -2],
                                   [-1, -1, -1, -1, -1, 0, 0.5, 0.5, -2, -2.5]])
        leigh_bear = pd.DataFrame(np.flip(np.array(leigh_bull), axis=0))
        cervelloroyo_bull = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                                          [0, 0, 0, -1, -2, -2, -2, -2, -2, -2],
                                          [0, 0, -1, -3, -3, -3, -3, -3, -3, -3],
                                          [0, -1, -3, -5, -5, -5, -5, -5, -5, -5],
                                          [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
                                          [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
                                          [5, -1, -5, -5, -5, -5, -5, -5, -5, -5]])
        cervelloroyo_bear = pd.DataFrame(np.flip(np.array(cervelloroyo_bull), axis=0))
        test7 = MatrixFlagLabels(close, 30, template_name='leigh_bear')
        test8 = MatrixFlagLabels(close, 30, template_name='cervelloroyo_bear')
        test9 = MatrixFlagLabels(close, 30, template_name='cervelloroyo_bull')
        pd.testing.assert_frame_equal(test7.template, leigh_bear)
        pd.testing.assert_frame_equal(test8.template, cervelloroyo_bear)
        pd.testing.assert_frame_equal(test9.template, cervelloroyo_bull)

        # Exception for invalid name.
        with self.assertRaises(Exception):
            MatrixFlagLabels(close[0:7], window=30, template_name='abcd')
        with self.assertRaises(Exception):
            test7._init_template('abcd')
