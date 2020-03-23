"""
This script implements the creation of dependence and distance matrix for the
methods of Information Codependence.
"""

#Imports
import numpy as np
import pandas as pd
import mlfinlab as ml
from sklearn.metrics import *
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage,fcluster
from mlfinlab.codependence import variation_of_information_score,get_mutual_info,distance_correlation
from mlfinlab.codependence.correlation import angular_distance,absolute_angular_distance,\
                                              squared_angular_distance,distance_correlation

def get_dependence_matrix(df : pd.DataFrame, dependence_method : str ):
    """
    This function returns a dependence matrix for the given method of dependence method.

    :param df: (pd.DataFrame) of features
    :param dependence_method: (str) the algorithm to be use for generating dependence_matrix,
                                either 'information_variation' or 'mutual_information' or 'distance_correlation' .
    :return: (pd.DataFrame)  of dependence_matrix
    """

    features_cols = df.columns.values #Get the feature names
    #Defining the dependence function
    if dependence_method == 'information_variation':
        dep_function = variation_of_information_score
    elif dependence_method == 'mutual_information':
        dep_function = get_mutual_info
    elif dependence_method == 'distance_correlation':
        dep_function = distance_correlation
    else:
        raise ValueError(f"{base_algorithm} is not a valid method. Use either 'information_variation' \
                            or 'mutual_information' or 'distance_correlation'.")
    #Generating the dependence_matrix for the defined method
    if dependence_method != 'distance_correlation':
        dependence_matrix = [[dep_function(df[x].values,df[y].values,normalize=True) for x in features_cols ] for y in features_cols]
    else:
        dependence_matrix = [[dep_function(df[x].values,df[y].values) for x in features_cols ] for y in features_cols]
    dependence_df = pd.DataFrame(data=dependence_matrix,index=features_cols,columns=features_cols) # dependence_matrix converted to a DataFrame
    if dependence_method == 'information_variation':
        return 1 - dependence_df  # IV is reverse, 1 - independent, 0 - similar
    else:
        return dependence_df


def get_distance_matrix(X:pd.DataFrame, distance_metric:str = 'angular'):
    """
    Apply distance operator to dependence matrix.

    :param X: (pd.DataFrame) to which distance operator to be applied.
    :param distance_metric: (str) the distance operator to be used for generating the distance matrix.
                                The methods that can be applied are -
                                'angular' , 'squared_angular' and 'absolute_angular'.
    :return: (pd.DataFrame) of distance matrix
    """
    if distance_metric == 'angular':
        distfun = lambda x: ((1 - x).round(5) / 2.) ** .5
    elif distance_metric == 'abs_angular':
        distfun = lambda x: ((1 - abs(x)).round(5) / 2.) ** .5
    elif distance_metric == 'squared_angular':
        distfun = lambda x: ((1 - x ** 2).round(5) / 2.) ** .5
    else:
        raise ValueError('Unknown distance metric')
    return distfun(X).fillna(0)

def get_feature_clusters(X : pd.DataFrame, dependence_metric:str, distance_metric:str, linkage_method : str , n_clusters : int = None):
    '''
    Get Clusters containing features subsets

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
    '''
    #Get the dependence metrix
    if dependence_metric != 'linear':
        dep_matrix = get_dependence_matrix(X , dependence_method = dependence_metric)
    else:
        dep_matrix = X.corr()
    #Apply distance operator on the dependence matrix
    dist_matrix = get_distance_matrix(dep_matrix)
    link = linkage(squareform(dist_matrix), method = linkage_method)

    if n_clusters is not None:
        if n_clusters >= len(X.columns):
            raise ValueError('Number of Clusters Must be less than the number of features')
        clusters = fcluster(link, t = n_clusters, criterion='maxclust')
        clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters+1)]
        return clustered_subsets

    else:
        n_clusters = len(ml.clustering.get_onc_clusters(dep_matrix)[1])
        clusters = fcluster(link, t = n_clusters, criterion='maxclust')
        clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters+1)]
        return clustered_subsets

def clustered_feature_importance(model, X, y, cv_gen, n_clusters=None, dependence_metric:str='information_variation', distance_metric:str='angular', linkage_method='single', sample_weight=None, scoring=log_loss):
    """
    Clustered Feature Importnance or Clustered MDA is the modified verison of MDA (Mean Decreased Accuracy). It is
    robust to substitution effect that takes place when two or more explanatory variables share a substantial amount of
    information (predictive power).
    First, apply a single linkage agglomerative clustering algorithm on a dependence matrix. Secondly, instead of shuffling (permutating)
    all variables individually (like in MDA), we shuffle all variables in cluster together. Next, we follow all the rest of the steps
    as in MDA.

    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param n_clusters: (int): number of clustered subsets of features to form,
                              if None then optimal number of clusters is decided by the ONC Algorithm.
    :param dependence_metric : (str) method to be use for generating dependence_matrix,
                                either 'linear' or 'information_variation' or 'mutual_information' or 'distance_correlation' .
                                Default is 'information_variation' which is robust to both linear and non-linear substitution effect.
    :param distance_metric: (str) the distance operator to be used for generating the distance matrix.
                                The methods that can be applied are -
                                'angular' , 'squared_angular' and 'absolute_angular'.
    :param linkage_method : (str) method of linkage to be used for clustering. Methods include -
                                  'single' , 'ward' , 'complete' , 'average' , 'weighted' and 'centroid'.
    :param sample_weight: (np.array): Sample weights, if None equal to ones.
    :param scoring: (function): Scoring function used to determine importance. Any sklearn scoring function.  Default is log_loss
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """
    #Generating the feature subsets
    clustered_subsets = get_feature_clusters(X, dependence_metric,distance_metric,linkage_method,n_clusters)

    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    fold_metrics_values, features_metrics_values = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        fit = model.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight[train])
        pred = fit.predict(X.iloc[test, :])
        # Get overall metrics value on out-of-sample fold
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            fold_metrics_values.loc[i] = -scoring(y.iloc[test], prob, sample_weight=sample_weight[test],
                                                  labels=model.classes_)
        else:
            fold_metrics_values.loc[i] = scoring(y.iloc[test], pred, sample_weight=sample_weight[test])
        # Get feature specific metric on out-of-sample fold
        for j in clustered_subsets: #Instead of Individual Features , clustered subsets are used
            X1_ = X.iloc[test, :].copy(deep=True)

            for j_i in j:
                np.random.shuffle(X1_[j_i].values)  # Permutation of a single column of the whole subsets
            if scoring == log_loss:
                prob = fit.predict_proba(X1_)
                features_metrics_values.loc[i, j] = -scoring(y.iloc[test], prob, sample_weight=sample_weight[test],
                                                             labels=model.classes_)
            else:
                pred = fit.predict(X1_)
                features_metrics_values.loc[i, j] = scoring(y.iloc[test], pred,
                                                            sample_weight=sample_weight[test])

    importance = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == log_loss:
        importance = importance / -features_metrics_values
    else:
        importance = importance / (1.0 - features_metrics_values)
    importance = pd.concat({'mean': importance.mean(), 'std': importance.std() * importance.shape[0] ** -.5}, axis=1)
    importance.replace([-np.inf, np.nan], 0, inplace=True)  # Replace infinite values

    return importance
