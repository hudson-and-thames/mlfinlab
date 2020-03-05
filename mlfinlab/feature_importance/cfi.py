"""
An Implementation of the Clustered Feature Importance algorithm described by
Dr Marcos Lopez de Prado in 'Clustered Feature Importance (Presentation Slides)'
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595)
"""

#Imports
import numpy as np
import pandas as pd
import mlfinlab as ml
from sklearn.metrics import log_loss
from scipy.cluster.hierarchy import linkage, fcluster
from information_matrix import  get_dependence_matrix , get_distance_matrix


def cluster_features(X : pd.DataFrame , linkage_method : str , n_clusters : int = None ):
    '''
    Get Clusters containing features subsets

    (param) X : (pd.DataFrame) of features
    (param) linkage_method : (str) method of linkage to be used for clustering. Methods include -
                                  'single' , 'ward' , 'complete' , 'average' , 'weighted' and 'centroid'.
    (param) C : (int) number of clusters to form.
                      Must be less the total number of features.
                      If None then it returns optimal number of clusters
                      decided by the ONC Algorithm.

    (return) : (array) of feature subsets
    '''
    vi_matrix = get_dependence_matrix(df =X , base_algorithm = 'variation_of_information')
    dist_matrix = get_distance_matrix(vi_matrix)
    link = linkage(dist_matrix, method = linkage_method)

    if n_clusters is not None:
        if n_clusters >= len(X.columns):
            raise ValueError('Number of Clusters Must be less than the number of features')
        clusters = fcluster(link, t = n_clusters, criterion='maxclust')
        clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters+1)]
        return clustered_subsets

    else:
        n_clusters = len(ml.clustering.get_onc_clusters(vi_matrix)[1])
        clusters = fcluster(link, t = n_clusters, criterion='maxclust')
        clustered_subsets = [[f for c, f in zip(clusters, X.columns) if c == ci] for ci in range(1, n_clusters+1)]
        return clustered_subsets

def clustered_feature_importance(model, X, y, cv_gen, n_clusters=None, linkage_method='single', sample_weight=None, scoring=log_loss):
    """
    Clustered Feature Importnance or Clustered MDA is modified verison of MDA (Mean Decreased Accuracy). It is very
    robust to substitution effect that takes place when two or more explanatory variables share a substantial amount of
    information (predictive power), while other feature importance method may struggle.

    First, apply a single linkage agglomerative clustering algorithm on variation of information (VI) matix, since VI can
    effectively measure linear and non-linear codependence. Secondly, instead of shuffling (permutating) all variables
    individually (like in MDA), we shuffle all variables in cluster together.Next, we follow all the rest of the steps
    as in MDA.

    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param n_clusters: (int): number of clustered subsets of features to form,
                              if None then optimal number of clusters is decided by the ONC Algorithm.
    :param linkage_method : (str) method of linkage to be used for clustering. Methods include -
                                  'single' , 'ward' , 'complete' , 'average' , 'weighted' and 'centroid'.
    :param sample_weight: (np.array): Sample weights, if None equal to ones.
    :param scoring: (function): Scoring function used to determine importance.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """

    clustered_subsets = cluster_features(X,linkage_method,n_clusters)

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
