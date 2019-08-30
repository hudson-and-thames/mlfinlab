"""
Module which implements feature importance algorithms described in Chapter 8
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from mlfinlab.cross_validation.cross_validation import ml_cross_val_score


# pylint: disable=invalid-name
# pylint: disable=invalid-unary-operand-type

def feature_importance_mean_decrease_impurity(clf, feature_names):
    """
    Snippet 8.2, page 115. MDI Feature importance

    This function generates feature importance from classifiers estimators importance
    using Mean Impurity Reduction (MDI) algorithm

    :param clf: (BaggingClassifier, RandomForest or any ensemble sklearn object): trained classifier
    :param feature_names: (list): array of feature names
    :return: (pd.DataFrame): individual MDI feature importance
    """
    # Feature importance based on IS mean impurity reduction
    feature_imp_df = {i: tree.feature_importances_ for i, tree in enumerate(clf.estimators_)}
    feature_imp_df = pd.DataFrame.from_dict(feature_imp_df, orient='index')
    feature_imp_df.columns = feature_names
    feature_imp_df = feature_imp_df.replace(0, np.nan)  # Because max_features = 1
    imp = pd.concat({'mean': feature_imp_df.mean(), 'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -.5},
                    axis=1)
    imp /= imp['mean'].sum()
    return imp


def feature_importance_mean_decrease_accuracy(clf, X, y, cv_gen, sample_weight=None, scoring='neg_log_loss'):
    """
    Snippet 8.3, page 116-117. MDA Feature Importance

    This function uses OOS performance to generate Mean Decrease Accuracy importance

    :param clf: (sklearn.ClassifierMixin): any sklearn classifier
    :param X: (pd.DataFrame): train set features
    :param y: (pd.DataFrame, np.array): train set labels
    :param cv_gen: (cross_validation.PurgedKFold): cross-validation object
    :param sample_weight: (np.array): sample weights, if None equal to ones
    :param scoring: (str): scoring function used to determine importance, either 'neg_log_loss' or 'accuracy'
    :return: (pd.DataFrame): mean and std feature importance
    """
    # Feature importance based on OOS score reduction
    scoring_func_dict = {'neg_log_loss': log_loss, 'accuracy': accuracy_score, 'f1': f1_score,
                         'precision': precision_score, 'recall': recall_score, 'roc_auc': roc_auc_score}
    try:
        scoring_func_dict[scoring]
    except KeyError:
        raise ValueError(
            'Invalid choice of scoring method. Possible scoring methods:{}.'.format(list(scoring_func_dict.keys())))

    scoring_func = scoring_func_dict[scoring]

    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    fold_metrics_values, features_metrics_values = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight[train])
        pred = fit.predict(X.iloc[test, :])

        # Get overall metrics value on out-of-sample fold
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            fold_metrics_values.loc[i] = -scoring_func(y.iloc[test], prob, sample_weight=sample_weight[test],
                                                       labels=clf.classes_)
        else:
            fold_metrics_values.loc[i] = scoring_func(y.iloc[test], pred, sample_weight=sample_weight[test])

        # Get feature specific metric on out-of-sample fold
        for j in X.columns:
            X1_ = X.iloc[test, :].copy(deep=True)
            np.random.shuffle(X1_[j].values)  # Permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                features_metrics_values.loc[i, j] = -scoring_func(y.iloc[test], prob, sample_weight=sample_weight[test],
                                                                  labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                features_metrics_values.loc[i, j] = scoring_func(y.iloc[test], pred,
                                                                 sample_weight=sample_weight[test])

    imp = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -features_metrics_values
    else:
        imp = imp / (1. - features_metrics_values)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    imp.replace([-np.inf, np.nan], 0, inplace=True)  # Replace infinite values
    return imp


def feature_importance_sfi(clf, X, y, cv_gen, sample_weight=None, scoring='neg_log_loss'):
    """
    Snippet 8.4, page 118. Implementation of SFI

    This function generates Single Feature Importance based on OOS score (using cross-validation object)

    :param clf: (sklearn.ClassifierMixin): any sklearn classifier
    :param X: (pd.DataFrame): train set features
    :param y: (pd.DataFrame, np.array): train set labels
    :param cv_gen: (cross_validation.PurgedKFold): cross-validation object
    :param sample_weight: (np.array): sample weights, if None equal to ones
    :param scoring: (str): scoring function used to determine importance
    :return: (pd.DataFrame): mean and std feature importance
    """
    feature_names = X.columns
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat in feature_names:
        feat_cross_val_scores = ml_cross_val_score(clf, X=X[[feat]], y=y, sample_weight=sample_weight,
                                                   scoring=scoring, cv_gen=cv_gen)
        imp.loc[feat, 'mean'] = feat_cross_val_scores.mean()
        imp.loc[feat, 'std'] = feat_cross_val_scores.std() * feat_cross_val_scores.shape[0] ** -.5
    return imp


def plot_feature_importance(imp, oob_score, oos_score, savefig=False, output_path=None):
    """
    Snippet 8.10, page 124. Feature importance plotting function

    Plot feature importance function
    :param imp: (pd.DataFrame): mean and std feature importance
    :param oob_score: (float): out-of-bag score
    :param oos_score: (float): out-of-sample (or cross-validation) score
    :param savefig: (bool): boolean flag to save figure to a file
    :param output_path: (str): if savefig is True, path where figure should be saved
    :return: None
    """
    # Plot mean imp bars with std
    plt.figure(figsize=(10, imp.shape[0] / 5))
    imp.sort_values('mean', ascending=True, inplace=True)
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('Feature importance. OOB Score:{}; OOS score:{}'.format(round(oob_score, 4), round(oos_score, 4)))
    if savefig is True:
        plt.savefig(output_path)
    else:
        plt.show()
