"""
Test Optimal Number of Clusters algorithm
"""

import unittest
import pandas as pd
from sklearn.datasets import load_breast_cancer

from mlfinlab.clustering import get_onc_clusters
from mlfinlab.clustering.onc import _check_improve_clusters


class TestOptimalNumberOfClusters(unittest.TestCase):
    """
    Test get_onc_clusters function
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        self.data, _ = load_breast_cancer(return_X_y=True)

    @staticmethod
    def _check_if_in_cluster(array, cluster_dict):
        """
        Check if array is in dictionary values not taking into account order of elements
        """
        array_set = set(array)
        flag = False
        for arr in cluster_dict.values():
            if set(arr) == array_set:
                flag = True
        return flag

    def test_get_onc_clusters(self):
        """
        Test get_onc_clusters function on Breast Cancer data set from sklearn
        """

        _, clusters, _ = get_onc_clusters(pd.DataFrame(self.data).corr(), repeat=50)
        self.assertGreaterEqual(len(clusters.keys()), 5)  # Optimal number of clusters
        self.assertTrue(self._check_if_in_cluster([11, 14, 18], clusters))  # Check clusters components
        self.assertTrue(self._check_if_in_cluster([0, 2, 3, 10, 12, 13, 20, 22, 23], clusters))
        self.assertTrue(self._check_if_in_cluster([5, 6, 7, 25, 26, 27], clusters))

    def test_check_redo_condition(self):
        """
        Function to test redo condition helper function
        """
        self.assertEqual((4, 5, 6), _check_improve_clusters(2, 3, (1, 2, 3), (4, 5, 6)))
