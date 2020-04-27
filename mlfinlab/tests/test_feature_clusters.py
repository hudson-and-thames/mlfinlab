"""
Test various methods of generating feature clusters
"""
import unittest

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
        self.X, self.y = get_classification_data(40, 5, 30, 1000, sigma=2)

    def test_get_feature_clusters(self):
        """
        Test get_feature_clusters arguments
        """
        #test for different dependence matrix

        clustered_subsets = get_feature_clusters(self.X, dependence_metric='information_variation',
                                                 distance_metric='angular', linkage_method='single',
                                                 n_clusters=2)
        #hierarchical auto clustering
        clustered_subsets_ha = get_feature_clusters(self.X, dependence_metric='linear',
                                                    distance_metric='angular', linkage_method='single',
                                                    n_clusters=None, critical_threshold=0.2)
        #test for optimal number of clusters and  _check_for_low_silhouette_scores
        #since this is done on test dataset so there will be no features with low silhouette score
        #so we will make a feature with some what lower silhouette score (near to zero) and set
        #the threshold higher (0.2) than that. Also we need a feature to trigger the low degree of freedom
        #condition so, we create a series of zero in the datasets
        self.X['R_5c'] = self.X['R_5'] #this feature is add to introduce low DF in the regressor.
        self.X['R_1c'] = self.X['R_1'] #this will trigger the expection of LinAlgError i.e. presence of singular matrix
        clustered_subsets_distance = get_feature_clusters(self.X, dependence_metric='linear',
                                                          distance_metric=None, linkage_method=None,
                                                          n_clusters=None, critical_threshold=0.2)

        #assertions
        #output clusters must be 2
        self.assertEqual(len(clustered_subsets), 2)
        #The ONC should detect somwhere around 5 clusters
        self.assertAlmostEqual(len(clustered_subsets_ha), 5, delta=1)
        self.assertAlmostEqual(len(clustered_subsets_distance), 5, delta=1)

    def test_value_error_raise(self):
        """
        Test get_feature_clusters for invalid number of clusters arguments
        """
        #Number of clusters larger than number of features
        with self.assertRaises(ValueError):
            get_feature_clusters(self.X, dependence_metric='linear',
                                 distance_metric='angular', linkage_method='single',
                                 n_clusters=int(41))
