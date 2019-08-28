.. _implementations-sb_bagging:

======================================================
Sequentially Bootstrapped Bagging Classifier/Regressor
======================================================

In sampling section we have shown that sampling should be done by Sequential Bootstrapping.
SequentiallyBootstrappedBaggingClassifier and SequentiallyBootstrappedBaggingRegressor extend sklearn's BaggingClassifier/Regressor by using Sequential Bootstrapping instead of random sampling.

In order to build indicator matrix, Triple Barrier Events and price bars used to label training data set should be passed as parameters.


Sequentially Bootstrapped Bagging Classifier
============================================

.. py:class:: SequentiallyBootstrappedBaggingClassifier(SequentiallyBootstrappedBaseBagging, BaggingClassifier,
                                                ClassifierMixin)
    events_end_times: pd.Series
          Triple-Barrier events used to label X_train, y_train. We need them for indicator matrix generation
          Expected columns are t1 (label endtime), index when label was started
      price_bars: pd.DataFrame
          Price bars used in events_end_times generation
      base_estimator : object or None, optional (default=None)
          The base estimator to fit on random subsets of the dataset.
          If None, then the base estimator is a decision tree.
      n_estimators : int, optional (default=10)
          The number of base estimators in the ensemble.
      max_samples : int or float, optional (default=1.0)
          The number of samples to draw from X to train each base estimator.
          - If int, then draw `max_samples` samples.
          - If float, then draw `max_samples * X.shape[0]` samples.
      max_features : int or float, optional (default=1.0)
          The number of features to draw from X to train each base estimator.
          - If int, then draw `max_features` features.
          - If float, then draw `max_features * X.shape[1]` features.
      bootstrap_features : boolean, optional (default=False)
          Whether features are drawn with replacement.
      oob_score : bool, optional (default=False)
          Whether to use out-of-bag samples to estimate
          the generalization error.
      warm_start : bool, optional (default=False)
          When set to True, reuse the solution of the previous call to fit
          and add more estimators to the ensemble, otherwise, just fit
          a whole new ensemble. See :term:`the Glossary <warm_start>`.
          .. versionadded:: 0.17
             *warm_start* constructor parameter.
      n_jobs : int or None, optional (default=None)
          The number of jobs to run in parallel for both `fit` and `predict`.
          ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
          ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
          for more details.
      random_state : int, RandomState instance or None, optional (default=None)
          If int, random_state is the seed used by the random number generator;
          If RandomState instance, random_state is the random number generator;
          If None, the random number generator is the RandomState instance used
          by `np.random`.
      verbose : int, optional (default=0)
          Controls the verbosity when fitting and predicting.


Sequentially Bootstrapped Bagging Regressor
============================================

.. py:class:: SequentiallyBootstrappedBaggingRegressor(SequentiallyBootstrappedBaseBagging, BaggingRegressor, RegressorMixin)
    events_end_times: pd.Series
          Triple-Barrier events used to label X_train, y_train. We need them for indicator matrix generation
          Expected columns are t1 (label endtime), index when label was started
      price_bars: pd.DataFrame
          Price bars used in events_end_times generation
      base_estimator : object or None, optional (default=None)
          The base estimator to fit on random subsets of the dataset.
          If None, then the base estimator is a decision tree.
      n_estimators : int, optional (default=10)
          The number of base estimators in the ensemble.
      max_samples : int or float, optional (default=1.0)
          The number of samples to draw from X to train each base estimator.
          - If int, then draw `max_samples` samples.
          - If float, then draw `max_samples * X.shape[0]` samples.
      max_features : int or float, optional (default=1.0)
          The number of features to draw from X to train each base estimator.
          - If int, then draw `max_features` features.
          - If float, then draw `max_features * X.shape[1]` features.
      bootstrap_features : boolean, optional (default=False)
          Whether features are drawn with replacement.
      oob_score : bool
          Whether to use out-of-bag samples to estimate
          the generalization error.
      warm_start : bool, optional (default=False)
          When set to True, reuse the solution of the previous call to fit
          and add more estimators to the ensemble, otherwise, just fit
          a whole new ensemble. See :term:`the Glossary <warm_start>`.
      n_jobs : int or None, optional (default=None)
          The number of jobs to run in parallel for both `fit` and `predict`.
          ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
          ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
          for more details.
      random_state : int, RandomState instance or None, optional (default=None)
          If int, random_state is the seed used by the random number generator;
          If RandomState instance, random_state is the random number generator;
          If None, the random number generator is the RandomState instance used
          by `np.random`.
      verbose : int, optional (default=0)
          Controls the verbosity when fitting and predicting.
      Attributes
      ----------
      estimators_ : list of estimators
          The collection of fitted sub-estimators.
      estimators_samples_ : list of arrays
          The subset of drawn samples (i.e., the in-bag samples) for each base
          estimator. Each subset is defined by an array of the indices selected.
      estimators_features_ : list of arrays
          The subset of drawn features for each base estimator.
      oob_score_ : float
          Score of the training dataset obtained using an out-of-bag estimate.
      oob_prediction_ : array of shape = [n_samples]
          Prediction computed with out-of-bag estimate on the training
          set. If n_estimators is small it might be possible that a data point
          was never left out during the bootstrap. In this case,
          `oob_prediction_` might contain NaN.
      References
      ----------
      .. [1] L. Breiman, "Pasting small votes for classification in large
             databases and on-line", Machine Learning, 36(1), 85-103, 1999.
      .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
             1996.
      .. [3] T. Ho, "The random subspace method for constructing decision
             forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
             1998.
      .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
             Learning and Knowledge Discovery in Databases, 346-361, 2012.
      """

An example of using SequentiallyBootstrappedBaggingRegressor:
::
  import pandas as pd
  from sklearn.ensemble import RandomForestClassifier
  from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier

  X = pd.read_csv('X_FILE_PATH', index_col=0, parse_dates = [0])
  y = pd.read_csv('y_FILE_PATH', index_col=0, parse_dates = [0])
  triple_barrier_events = pd.read_csv('BARRIER_FILE_PATH', index_col=0, parse_dates = [0, 2])
  price_bars = pd.read_csv('PRICE_BARS_FILE_PATH', index_col=0, parse_dates = [0, 2])

  triple_barrier_events = triple_barrier_events.loc[X.index, :] # take only train part
  price_events = price_events[(price_events.index >= X.index.min()) & (price_events.index <= X.index.max())]

  base_est = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                   class_weight='balanced_subsample')
  clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=base_est, events_end_times=triple_barrier_events.t1,
                                                  price_bars=price_bars, oob_score=True)
  clf.fit(X, y)
