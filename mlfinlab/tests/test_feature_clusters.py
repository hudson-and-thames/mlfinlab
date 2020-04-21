"""
Test various methods of generating feature clusters
"""
import unittest
import pandas as pd
from sklearn.datasets import make_classification

from mlfinlab.clustering.feature_clusters import get_feature_clusters

def getTestData(n_features=100,n_informative=25,n_redundant=25,n_samples=10000,random_state=0,sigmaStd=.0):
    '''
    A funtion to generate synthetic data
    '''
    np.random.seed(random_state)
    X,y=make_classification(n_samples=n_samples,n_features=n_features-n_redundant, n_informative=n_informative,
                            n_redundant=0,shuffle=False,random_state=random_state)
    cols=['I_'+str(i) for i in range(n_informative)]
    cols+=['N_'+str(i) for i in range(n_features-n_informative-n_redundant)]
    X,y=pd.DataFrame(X,columns=cols),pd.Series(y)
    i=np.random.choice(range(n_informative),size=n_redundant)
    for k,j in enumerate(i):
        X['R_'+str(k)]=X['I_'+str(j)]+np.random.normal(size=X.shape[0])*sigmaStd
    return X,y

class TestFeatureClusters(unittest.TestCase):
    """
    Test get_feature_clusters function
    """

    def setUp(self):
        """
        Create X, y datasets
        """
        self.X, self.y = getTestData(40,5,30,10000,sigmaStd=.125)

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
        #the threshold higher (0.55) than that. Also we need a feature to trigger the low degree of freedom
        #condition so, we create a series of zero in the datasets
        X_data = self.X
        X_data['low_df'] = 0
        clustered_subsets_distance = get_feature_clusters(X_data, dependence_metric='distance_correlation',
                                                          distance_metric='abs_angular', linkage_method='single',
                                                          n_clusters=None, critical_threshold=0.55)
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
            get_feature_clusters(selfXa, dependence_metric='linear',
                                 distance_metric='angular', linkage_method='single',
                                 n_clusters=int(len(self.data)))
