"""
An Implementation of the Clustered Feature Importance algorithm described by
Dr Marcos Lopez de Prado in 'Clustered Feature Importance (Presentation Slides)'
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595)
"""
#Imports
import numpy as np
import pandas as pd
from sklearn.metrics import *


def clustered_feature_importance(model, X, y, cv_gen, clustered_subsets: list, sample_weight_train: np.array = None, sample_weight_score: np.array = None, scoring=log_loss):
    """
    Clustered Feature Importance or Clustered MDA is the modified verison of MDA (Mean Decreased Accuracy). It is
    robust to substitution effect that takes place when two or more explanatory variables share a substantial amount of
    information (predictive power).
    First, apply a single linkage agglomerative clustering algorithm on a dependence matrix. Secondly, instead of shuffling (permutating)
    all variables individually (like in MDA), we shuffle all variables in cluster together. Next, we follow all the rest of the steps
    as in MDA.
    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param clustered_subsets: (list) of feature clusters
    :param sample_weight_train: (np.array): Sample weights, if None equal to ones.
    :param sample_weight_score: (np.array): Sample weights used to evaluate the model
    :param scoring: (function): Scoring function used to determine importance. Any sklearn scoring function.  Default is log_loss
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))
    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))
    fold_metrics_values, features_metrics_values = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        fit = model.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight_train[train])
        pred = fit.predict(X.iloc[test, :])
        # Get overall metrics value on out-of-sample fold
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            fold_metrics_values.loc[i] = -scoring(y.iloc[test], prob, sample_weight=sample_weight_score[test],
                                                  labels=model.classes_)
        else:
            fold_metrics_values.loc[i] = scoring(y.iloc[test], pred, sample_weight=sample_weight_score[test])
        # Get feature specific metric on out-of-sample fold
        for j in clustered_subsets:  # Instead of Individual Features , clustered subsets are used
            X1_ = X.iloc[test, :].copy(deep=True)
            for j_i in j:
                np.random.shuffle(X1_[j_i].values)  # Permutation of a single column of the whole subsets
        if scoring == log_loss:
            prob = fit.predict_proba(X1_)
            features_metrics_values.loc[i, j] = -scoring(y.iloc[test], prob, sample_weight=sample_weight_score[test],
                                                         labels=model.classes_)
        else:
            pred = fit.predict(X1_)
            features_metrics_values.loc[i, j] = scoring(y.iloc[test], pred,
                                                        sample_weight=sample_weight_score[test])
    importance = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == log_loss:
        importance = importance / -features_metrics_values
    else:
        importance = importance / (1.0 - features_metrics_values)
    importance = pd.concat({'mean': importance.mean(), 'std': importance.std() * importance.shape[0] ** -.5}, axis=1)
    importance.replace([-np.inf, np.nan], 0, inplace=True)  # Replace infinite values
    return importance
