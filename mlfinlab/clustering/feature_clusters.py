"""
This implementation creates clustered subsets of features described in the paper
Clustered Feature Importance (Presentation Slides) by Dr Marcos Lopez de Prado
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595
"""
#Imports
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from mlfinlab.clustering import get_onc_clusters
from mlfinlab.codependence.codependence_matrix import get_dependence_matrix, get_distance_matrix

def get_feature_clusters(X: pd.DataFrame, dependence_metric: str, distance_metric: str, linkage_method: str, n_clusters: int = None):
    """
    Get clustered features subsets from the given set of features.
    :param X : (pd.DataFrame) of features
    :param dependence_metric : (str) method to be use for generating dependence_matrix,
                                either 'linear' or 'information_variation' or 'mutual_information' or 'distance_correlation' .
    :param distance_metric: (str) the distance operator to be used for generating the distance matrix.
                                The methods that can be applied are -
                                'angular' , 'squared_angular' and 'absolute_angular'.
    :param linkage_method : (str) method of linkage to be used for clustering. Methods include -
                                  'single' , 'ward' , 'complete' , 'average' , 'weighted' and 'centroid'.
    :param n_clusters : (int) number of clusters to form. Must be less the total number of features.
                      If None then it returns optimal number of clusters decided by the ONC Algorithm.
    (return) : (array) of feature subsets
    """
    # Get the dependence matrix
    if dependence_metric != 'linear':
        dep_matrix = get_dependence_matrix(X, dependence_method=dependence_metric)
    else:
        dep_matrix = X.corr()
    # Apply distance operator on the dependence matrix
    dist_matrix = get_distance_matrix(dep_matrix, distance_metric=distance_metric)
    link = linkage(squareform(dist_matrix), method=linkage_method)
    if n_clusters is None:
        n_clusters = len(get_onc_clusters(dep_matrix.fillna(0))[1])# Get optimal number of clusters
    if n_clusters >= len(X.columns): #Check if number of clusters exceeds number of features
        raise ValueError('Number of clusters must be less than the number of features')
    clusters = fcluster(link, t=n_clusters, criterion='maxclust')
    clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters + 1)]
    return clustered_subsets
