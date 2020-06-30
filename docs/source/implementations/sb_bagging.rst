.. _implementations-sb_bagging:

======================================================
Sequentially Bootstrapped Bagging Classifier/Regressor
======================================================

In sampling section we have shown that sampling should be done by Sequential Bootstrapping.
SequentiallyBootstrappedBaggingClassifier and SequentiallyBootstrappedBaggingRegressor extend `sklearn <https://scikit-learn.org/>`_'s
BaggingClassifier/Regressor by using Sequential Bootstrapping instead of random sampling.

In order to build indicator matrix we need Triple Barrier Events (samples_info_sets) and price bars used to label
training data set. That is why samples_info_sets and price bars are input parameters for classifier/regressor.

Implementation
##############

.. py:currentmodule:: mlfinlab.ensemble.sb_bagging
.. automodule:: mlfinlab.ensemble.sb_bagging
   :members: SequentiallyBootstrappedBaggingClassifier, SequentiallyBootstrappedBaggingRegressor

Example
#######

An example of using SequentiallyBootstrappedBaggingClassifier

.. code-block::

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from mlfinlab.ensemble import SequentiallyBootstrappedBaggingClassifier

    X = pd.read_csv('X_FILE_PATH', index_col=0, parse_dates = [0])
    y = pd.read_csv('y_FILE_PATH', index_col=0, parse_dates = [0])
    triple_barrier_events = pd.read_csv('BARRIER_FILE_PATH', index_col=0, parse_dates = [0, 2])
    price_bars = pd.read_csv('PRICE_BARS_FILE_PATH', index_col=0, parse_dates = [0, 2])

    triple_barrier_events = triple_barrier_events.loc[X.index, :] # take only train part
    price_events = price_events[(price_events.index >= X.index.min()) &
                                (price_events.index <= X.index.max())]

    base_est = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                   class_weight='balanced_subsample')
    clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=base_est,
                                                    samples_info_sets=triple_barrier_events.t1,
                                                    price_bars=price_bars, oob_score=True)
    clf.fit(X, y)
