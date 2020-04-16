"""
Test various methods of generating feature clusters
"""
import unittest
import pandas as pd
from sklearn.datasets import load_breast_cancer

from mlfinlab.clustering.feature_clusters import get_feature_clusters

class TestFeatureClusters(unittest.TestCase):
    """
    Test get_feature_clusters function
    """

    def setUp(self):
        """
        Load a sample dataset
        """
        dataset, _ = load_breast_cancer(return_X_y=True)
        self.data = pd.DataFrame(dataset)

    def test_get_feature_clusters(self):
        """
        Test get_feature_clusters arguments
        """
        clustered_subsets_distance = get_feature_clusters(self.data, dependence_metric='distance_correlation',
                                                          distance_metric='abs_angular', linkage_method='single',
                                                          n_clusters=2)
        clustered_subsets_vi = get_feature_clusters(self.data, dependence_metric='information_variation',
                                                    distance_metric='squared_angular', linkage_method='single',
                                                    n_clusters=2)
        clustered_subsets_mi = get_feature_clusters(self.data, dependence_metric='mutual_information',
                                                    distance_metric='angular', linkage_method='single',
                                                    n_clusters=2)
        #output clusters must be 2
        self.assertAlmostEqual(len(clustered_subsets_distance), 2, delta=0.001)
        self.assertAlmostEqual(len(clustered_subsets_vi), 2, delta=0.001)
        self.assertAlmostEqual(len(clustered_subsets_mi), 2, delta=0.001)

    def test_value_error_raise(self):
        """
        Test get_feature_clusters , codependence_matrix and distance_matrix for invalid arguments
        """
        #Unkown dependence_metric
        with self.assertRaises(ValueError):
            get_feature_clusters(self.data, dependence_metric='information',
                                 distance_metric='angular', linkage_method='single',
                                 n_clusters=2)
        #Unkown distance_metric
        with self.assertRaises(ValueError):
            get_feature_clusters(self.data, dependence_metric='linear',
                                 distance_metric='serial', linkage_method='single',
                                 n_clusters=2)
        #Number of clusters larger than number of features
        with self.assertRaises(ValueError):
            get_feature_clusters(self.data, dependence_metric='linear',
                                 distance_metric='angular', linkage_method='single',
                                 n_clusters=int(len(self.data)))
