.. _implementations-sb_bagging:

======================================================
Sequentially Bootstrapped Bagging Classifier/Regressor
======================================================

In sampling section we have shown that sampling should be done by Sequential Bootstrapping.
SequentiallyBootstrappedBaggingClassifier and SequentiallyBootstrappedBaggingRegressor extend sklearn's BaggingClassifier/Regressor by using Sequential Bootstrapping instead of random sampling.


Sequentially Bootstrapped Bagging Classifier
============================================

.. py:class:: SequentiallyBootstrappedBaggingClassifier(SequentiallyBootstrappedBaseBagging, BaggingClassifier,
                                                ClassifierMixin)
    triple_barrier_events: pd.DataFrame
          Triple-Barrier events used to label X_train, y_train. We need them for indicator matrix generation
      price_bars: pd.DataFrame
          Price bars used in triple_barrier_events generation
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
