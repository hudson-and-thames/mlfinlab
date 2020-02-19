"""
Module which implements feature importance algorithms as described in Chapter 8 of Advances in Financial Machine Learning.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from mlfinlab.cross_validation.cross_validation import ml_cross_val_score


# pylint: disable=invalid-name
# pylint: disable=invalid-unary-operand-type
# pylint: disable=comparison-with-callable

def mean_decrease_impurity(model, feature_names):
    """
    Snippet 8.2, page 115. MDI Feature importance

    Mean decrease impurity (MDI) is a fast, explanatory-importance (in-sample, IS) method specific to tree-based
    classifiers, like RF. At each node of each decision tree, the selected feature splits the subset it received in
    such a way that impurity is decreased. Therefore, we can derive for each decision tree how much of the overall
    impurity decrease can be assigned to each feature. And given that we have a forest of trees, we can average those
    values across all estimators and rank the features accordingly.

    Tip:
    Masking effects take place when some features are systematically ignored by tree-based classifiers in favor of
    others. In order to avoid them, set max_features=int(1) when using sklearn’s RF class. In this way, only one random
    feature is considered per level.

    Notes:

    * MDI cannot be generalized to other non-tree based classifiers
    * The procedure is obviously in-sample.
    * Every feature will have some importance, even if they have no predictive power whatsoever.
    * MDI has the nice property that feature importances add up to 1, and every feature importance is bounded between 0 and 1.
    * method does not address substitution effects in the presence of correlated features. MDI dilutes the importance of
      substitute features, because of their interchangeability: The importance of two identical features will be halved,
      as they are randomly chosen with equal probability.
    * Sklearn’s RandomForest class implements MDI as the default feature importance score. This choice is likely
      motivated by the ability to compute MDI on the fly, with minimum computational cost.

    :param model: (model object): Trained tree based classifier.
    :param feature_names: (list): Array of feature names.
    :return: (pd.DataFrame): Mean and standard deviation feature importance.
    """
    # Feature importance based on in-sample (IS) mean impurity reduction
    feature_imp_df = {i: tree.feature_importances_ for i, tree in enumerate(model.estimators_)}
    feature_imp_df = pd.DataFrame.from_dict(feature_imp_df, orient='index')
    feature_imp_df.columns = feature_names

    # Make sure that features with zero importance are not averaged, since the only reason for a 0 is that the feature
    # was not randomly chosen. Replace those values with np.nan
    feature_imp_df = feature_imp_df.replace(0, np.nan)  # Because max_features = 1

    importance = pd.concat({'mean': feature_imp_df.mean(),
                            'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -0.5},
                           axis=1)
    importance /= importance['mean'].sum()
    return importance


def mean_decrease_accuracy(model, X, y, cv_gen, sample_weight=None, scoring=log_loss):
    """
    Snippet 8.3, page 116-117. MDA Feature Importance

    Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method. First, it fits a
    classifier; second, it derives its performance OOS according to some performance score (accuracy, negative log-loss,
    etc.); third, it permutates each column of the features matrix (X), one column at a time, deriving the performance
    OOS after each column’s permutation. The importance of a feature is a function of the loss in performance caused by
    its column’s permutation. Some relevant considerations include:

    * This method can be applied to any classifier, not only tree-based classifiers.
    * MDA is not limited to accuracy as the sole performance score. For example, in the context of meta-labeling
      applications, we may prefer to score a classifier with F1 rather than accuracy. That is one reason a better
      descriptive name would have been “permutation importance.” When the scoring function does not correspond to a
      metric space, MDA results should be used as a ranking.
    * Like MDI, the procedure is also susceptible to substitution effects in the presence of correlated features.
      Given two identical features, MDA always considers one to be redundant to the other. Unfortunately, MDA will make
      both features appear to be outright irrelevant, even if they are critical.
    * Unlike MDI, it is possible that MDA concludes that all features are unimportant. That is because MDA is based on
      OOS performance.
    * The CV must be purged and embargoed.


    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param sample_weight: (np.array): Sample weights, if None equal to ones.
    :param scoring: (function): Scoring function used to determine importance.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """

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
        for j in X.columns:
            X1_ = X.iloc[test, :].copy(deep=True)
            np.random.shuffle(X1_[j].values)  # Permutation of a single column
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


def single_feature_importance(clf, X, y, cv_gen, sample_weight=None, scoring=log_loss):
    """
    Snippet 8.4, page 118. Implementation of SFI

    Substitution effects can lead us to discard important features that happen to be redundant. This is not generally a
    problem in the context of prediction, but it could lead us to wrong conclusions when we are trying to understand,
    improve, or simplify a model. For this reason, the following single feature importance method can be a good
    complement to MDI and MDA.

    Single feature importance (SFI) is a cross-section predictive-importance (out-of- sample) method. It computes the
    OOS performance score of each feature in isolation. A few considerations:

    * This method can be applied to any classifier, not only tree-based classifiers.
    * SFI is not limited to accuracy as the sole performance score.
    * Unlike MDI and MDA, no substitution effects take place, since only one feature is taken into consideration at a time.
    * Like MDA, it can conclude that all features are unimportant, because performance is evaluated via OOS CV.

    The main limitation of SFI is that a classifier with two features can perform better than the bagging of two
    single-feature classifiers. For example, (1) feature B may be useful only in combination with feature A;
    or (2) feature B may be useful in explaining the splits from feature A, even if feature B alone is inaccurate.
    In other words, joint effects and hierarchical importance are lost in SFI. One alternative would be to compute the
    OOS performance score from subsets of features, but that calculation will become intractable as more features are
    considered.

    :param clf: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param sample_weight: (np.array): Sample weights, if None equal to ones.
    :param scoring: (function): Scoring function used to determine importance.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """
    feature_names = X.columns
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat in feature_names:
        feat_cross_val_scores = ml_cross_val_score(clf, X=X[[feat]], y=y, sample_weight=sample_weight,
                                                   scoring=scoring, cv_gen=cv_gen)
        imp.loc[feat, 'mean'] = feat_cross_val_scores.mean()
        # pylint: disable=unsubscriptable-object
        imp.loc[feat, 'std'] = feat_cross_val_scores.std() * feat_cross_val_scores.shape[0] ** -.5
    return imp


def plot_feature_importance(importance_df, oob_score, oos_score, save_fig=False, output_path=None):
    """
    Snippet 8.10, page 124. Feature importance plotting function.

    Plot feature importance.

    :param importance_df: (pd.DataFrame): Mean and standard deviation feature importance.
    :param oob_score: (float): Out-of-bag score.
    :param oos_score: (float): Out-of-sample (or cross-validation) score.
    :param save_fig: (bool): Boolean flag to save figure to a file.
    :param output_path: (str): If save_fig is True, path where figure should be saved.
    """
    # Plot mean imp bars with std
    plt.figure(figsize=(10, importance_df.shape[0] / 5))
    importance_df.sort_values('mean', ascending=True, inplace=True)
    importance_df['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=importance_df['std'], error_kw={'ecolor': 'r'})
    plt.title('Feature importance. OOB Score:{}; OOS score:{}'.format(round(oob_score, 4), round(oos_score, 4)))

    if save_fig is True:
        plt.savefig(output_path)
    else:
        plt.show()
