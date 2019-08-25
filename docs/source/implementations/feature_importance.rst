.. _implementations-feature_importance:

==================
Feature importance
==================

One of the key research principles of Advances in Financial Machine learning is:

`Backtesting is not a research tool. Feature importance is.`

There are three ways to get feature importance scores:

1) Mean Decrease Impurity (MDI). This score can be obtained from tree-based classifiers and corresponds to sklearn's feature_importances_ attribute. MDI uses in-sample (IS) performance to estimate feature importance.
2) Mean Decrease Accuracy (MDA). This method can be applied to any tree-based classifier, not only tree based. MDA uses out-of-sample (OOS) performance in order to estimate feature importance.
3) Single Feature Importance (SFI). MDA and MDI feature suffer from substitution effects: if two feature are highly correlated, one of them will be considered as important while the other one will be redundant.
                                    SFI is OOS feature importance estimator which doesn't suffer from substitution effect because it estimates each feature importance separately.

MDI, MDA, SFI feature importance
================================


.. py:function:: feature_importance_mean_imp_reduction(clf, feature_names)

    Snippet 8.2, page 115. MDI Feature importance

    This function generates feature importance from classifiers estimators importance
    using Mean Impurity Reduction (MDI) algorithm

    :param clf: (BaggingClassifier, RandomForest or any ensemble sklearn object): trained classifier
    :param feature_names: (list): array of feature names
    :return: (pd.DataFrame): individual MDI feature importance

.. py:function:: feature_importance_mean_decrease_accuracy(clf, X, y, cv_gen, sample_weight=None, scoring='neg_log_loss')

    Snippet 8.3, page 116-117. MDA Feature Importance

    :param clf: (sklearn.ClassifierMixin): any sklearn classifier
    :param X: (pd.DataFrame): train set features
    :param y: (pd.DataFrame, np.array): train set labels
    :param cv_gen: (PurgedKFold): cross-validation object
    :param sample_weight: (np.array): sample weights, if None equal to ones
    :param scoring: (str): scoring function used to determine importance, either 'neg_log_loss' or 'accuracy'
    :return: (pd.DataFrame): mean and std feature importance

.. py:function:: feature_importance_sfi(clf, X, y, cv_gen, sample_weight=None, scoring='neg_log_loss')

    Snippet 8.4, page 118. Implementation of SFI

    :param clf: (sklearn.ClassifierMixin): any sklearn classifier
    :param X: (pd.DataFrame): train set features
    :param y: (pd.DataFrame, np.array): train set labels
    :param cv_gen: (PurgedKFold): cross-validation object
    :param sample_weight: (np.array): sample weights, if None equal to ones
    :param scoring: (str): scoring function used to determine importance, either 'neg_log_loss' or 'accuracy'
    :return: (pd.DataFrame): mean and std feature importance

.. py:function:: plot_feature_importance(imp, oob_score, oos_score, savefig=False, output_path=None)

    Snippet 8.10, page 124. Feature importance plotting function

    Plot feature importance function
    :param imp: (pd.DataFrame): mean and std feature importance
    :param oob_score: (float): out-of-bag score
    :param oos_score: (float): out-of-sample (or cross-validation) score
    :param savefig: (bool): boolean flag to save figure to a file
    :param output_path: (str): if savefig is True, path where figure should be saved
    :return: None


An example showing how to use various feature importance functions::

  import pandas as pd
  from sklearn.ensemble import RandomForestClassifier
  from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier
	from mlfinlab.feature_importance import feature_importance_mean_imp_reduction, feature_importance_mean_decrease_accuracy,
                                          feature_importance_sfi, plot_feature_importance
  from mlfinlab.cross_validation import PurgedKFold, ml_cross_val_score
  from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier


  X_train = pd.read_csv('X_FILE_PATH', index_col=0, parse_dates = [0])
  y_train = pd.read_csv('y_FILE_PATH', index_col=0, parse_dates = [0])
  triple_barrier_events = pd.read_csv('BARRIER_FILE_PATH', index_col=0, parse_dates = [0, 2])
  price_bars = pd.read_csv('PRICE_BARS_FILE_PATH', index_col=0, parse_dates = [0, 2])

  triple_barrier_events = triple_barrier_events.loc[X.index, :] # take only train part
  price_events = price_events[(price_events.index >= X.index.min()) & (price_events.index <= X.index.max())]

  cv_gen = PurgedKFold(n_splits=4, info_sets=triple_barrier_events.t1)

  base_est = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                   class_weight='balanced_subsample')
  clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=base_est, triple_barrier_events=triple_barrier_events,
                                                  price_bars=price_bars, oob_score=True)
  clf.fit(X_train, y_train)

  oos_score = ml_cross_val_score(sclf, X_train, y_train, cv_gen=cv_gen, sample_weight=None,
                                       scoring='accuracy').mean()

  mdi_feature_imp = feature_importance_mean_imp_reduction(clf, X_train.columns)
  mda_feature_imp = feature_importance_mean_decrease_accuracy(clf, X_train, y_train, cv_gen, scoring='neg_log_loss')
  sfi_feature_imp = feature_importance_sfi(clf, X_train, y_train, cv_gen, scoring='accuracy')

  plot_feature_importance(mdi_feat_imp, oob_score=clf.oob_score_, oos_score=oos_score,
                                savefig=True, output_path='mdi_feat_imp.png')
  plot_feature_importance(mda_feat_imp, oob_score=clf.oob_score_, oos_score=oos_score,
                                savefig=True, output_path='mda_feat_imp.png')
  plot_feature_importance(sfi_feat_imp, oob_score=clf.oob_score_, oos_score=oos_score,
                                savefig=True, output_path='mdi_feat_imp.png')

Resulting images:

.. image:: feature_imp_images/mdi_feat_imp.png
   :scale: 100 %
   :align: center

.. image:: feature_imp_images/mda_feat_imp.png
  :scale: 100 %
  :align: center

.. image:: feature_imp_images/sfi_feat_imp.png
   :scale: 100 %
   :align: center
