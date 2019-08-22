"""
Module which implements feature importance algorithms described in Chapter 8
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import matplotlib as mpl
from mlfinlab.util import mp_pandas_obj
from mlfinlab.cross_validation.cross_validation import PurgedKFold, ml_cross_val_score


def feature_importance_mean_imp_reduction(clf, feature_names):
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


def feature_importance_mean_decrease_accuracy(clf, X, y, triple_barrier_events, n_splits=3, sample_weight=None,
                                              pct_embargo=0.,
                                              scoring='neg_log_loss'):
    """
    Snippet 8.3, page 116-117. MDA Feature Importance

    :param clf: (BaggingClassifier, RandomForest or any ensemble sklearn object): trained classifier
    :param X: (pd.DataFrame): train set features
    :param y: (pd.DataFrame, np.array): train set labels
    :param triple_barrier_events: (pd.Series): Triple-Barrier-Events used to sample training set (t1 and index is needed)
    :param n_splits: (int): the number of splits, default to 3
    :param sample_weight: (np.array): sample weights, if None equal to ones
    :param pct_embargo: (float): percent that determines the embargo size
    :param scoring: (str): scoring function used to determine importance, either 'neg_log_loss' or 'accuracy'
    :return: (pd.DataFrame): mean and std feature importance
    """
    # Feature importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise ValueError('wrong scoring method.')

    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    cv_gen = PurgedKFold(n_splits=n_splits, info_sets=triple_barrier_events.t1, pct_embargo=pct_embargo)  # Purged cv
    fold_metrics_values, features_metrics_values = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0)
        pred = fit.predict(X1)

        # Get overall metrics value on out-of-sample fold
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            fold_metrics_values.loc[i] = -log_loss(y1, prob, sample_weight=w1,
                                                   labels=clf.classes_)
        elif scoring == 'accuracy_score':
            fold_metrics_values.loc[i] = accuracy_score(y1, pred, sample_weight=w1)

        # Get feature specific metric on out-of-sample fold
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # Permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                features_metrics_values.loc[i, j] = -log_loss(y1, prob, sample_weight=w1, labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                features_metrics_values.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1)

    imp = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -features_metrics_values
    else:
        imp = imp / (1. - features_metrics_values)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    return imp


def feature_importance_sfi(clf, X, y, sample_weight=None, scoring='neg_log_loss', cv_gen=None):
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


def plot_feature_importance(imp, oob, oos, method, tag=0, simNum=0, savefig=False, output_path=None):
    # Plot mean imp bars with std
    mpl.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'],
                          error_kw={'ecolor': 'r'})
    if method == 'MDI':
        mpl.xlim([0, imp.sum(axis=1).max()])
        mpl.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index): ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center',
                                                    va='center',
                                                    color='black')
    mpl.title('tag = ' + tag + ' | simNum = ' + str(simNum) + ' | oob = ' + str(round(oob, 4)) +
              ' | oos = ' + str(round(oos, 4)))
    if savefig is True:
        mpl.savefig(output_path + 'featImportance_' + str(simNum) + '.png', dpi=100)
        mpl.clf()
        mpl.close()
    else:
        mpl.show()
    return
