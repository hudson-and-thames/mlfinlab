"""
Test various methods of generating feature clusters
"""
import unittest
import pandas as pd

from mlfinlab.util.generate_dataset import get_classification_data
from mlfinlab.clustering.feature_clusters import get_feature_clusters


# pylint: disable=invalid-name
class TestFeatureClusters(unittest.TestCase):
    """
    Test get_feature_clusters function
    """

    def setUp(self):
        """
        Create X, y datasets
        """
        self.X, self.y = get_classification_data(40, 5, 30, 1000, sigmaStd=2)

    def test_get_feature_clusters(self):
        """
        Test get_feature_clusters arguments
        """
        #test for different dependence matrix

        clustered_subsets_vi = get_feature_clusters(self.X, dependence_metric='information_variation',
                                                    distance_metric='squared_angular', linkage_method='single',
                                                    n_clusters=2)
        clustered_subsets_mi = get_feature_clusters(self.X, dependence_metric='mutual_information',
                                                    distance_metric='angular', linkage_method='single',
                                                    n_clusters=2)
        #test for optimal number of clusters and  _check_for_low_silhouette_scores
        #since this is done on test dataset so there will be no features with low silhouette score
        #so we will make a feature with some what lower silhouette score (near to zero) and set
        #the threshold higher (0.1) than that. Also we need a feature to trigger the low degree of freedom
        #condition so, we create a series of zero in the datasets
        self.X['R_5c'] = self.X['R_5'] #this feature is add to introduce low DF in the regressor.
        clustered_subsets_distance = get_feature_clusters(X_data, dependence_metric='distance_correlation',
                                                          distance_metric='abs_angular', linkage_method='single',
                                                          n_clusters=None, critical_threshold=0.1)
        #output clusters must be 2
        self.assertAlmostEqual(len(clustered_subsets_distance), 5, delta=0.001)
        self.assertAlmostEqual(len(clustered_subsets_vi), 2, delta=0.001)
        self.assertAlmostEqual(len(clustered_subsets_mi), 2, delta=0.001)


    def test_value_error_raise(self):
        """
        Test get_feature_clusters , codependence_matrix and distance_matrix for invalid arguments
        """
        #Unkown dependence_metric
        with self.assertRaises(ValueError):
            get_feature_clusters(self.X, dependence_metric='information',
                                 distance_metric='angular', linkage_method='single',
                                 n_clusters=2)
        #Unkown distance_metric
        with self.assertRaises(ValueError):
            get_feature_clusters(self.X, dependence_metric='linear',
                                 distance_metric='serial', linkage_method='single',
                                 n_clusters=2)
        #Number of clusters larger than number of features
        with self.assertRaises(ValueError):
            get_feature_clusters(self.X, dependence_metric='linear',
                                 distance_metric='angular', linkage_method='single',
                                 n_clusters=int(len(self.data)))
