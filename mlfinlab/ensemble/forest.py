"""
Implementing scikit-learn RandomForest/ExtraTrees Classifiers/Regressors with Sequential Bootstrapping
"""

import threading
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from abc import ABCMeta, abstractmethod

from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack

from sklearn.base import ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.ensemble.base import BaseEnsemble, _partition_estimators
from sklearn.ensemble.forest import _accumulate_prediction
from sklearn.utils.fixes import parallel_helper, _joblib_parallel_args
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.exceptions import DataConversionWarning
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier, ExtraTreeRegressor)
from sklearn.metrics import r2_score

MAX_INT = np.iinfo(np.int32).max


class SequentialBaseForest(BaseEnsemble, MultiOutputMixin, metaclass=ABCMeta):
    """
    Sequential Bootstrapping extension of BaseForest from sklearn (https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py)
    BaseForest class is authored by:
        Gilles Louppe <g.louppe@gmail.com>
        Brian Holt <bdholt1@gmail.com>
        Joly Arnaud <arnaud.v.joly@gmail.com>
        Fares Hedayati <fares.hedayati@gmail.com>
        License: BSD 3 clause
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

    def apply(self, X):
        """
        apply function from BaseForest sklearn
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py
        """

        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           **_joblib_parallel_args(prefer="threads"))(
            delayed(parallel_helper)(tree, 'apply', X, check_input=False)
            for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """
        decision path function from sklearn
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py
        """
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                              **_joblib_parallel_args(prefer='threads'))(
            delayed(parallel_helper)(tree, 'decision_path', X,
                                     check_input=False)
            for tree in self.estimators_)

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def generate_sample_indices(self, bootstrapped_events, prob, random_state):
        """
        This function applies Sequential Bootstrapping to generate samples for ensemble models.
        Uses mlfinlab.sampling.bootstrapping import seq_bootstrap which implements Sequential Bootstrapping described in
        the book.
        :param bootstrapped_events: (list): array of bootstrapped random indices
        :param prob: (list): probability of drawing corresponding sample array from bootstrapped_events
        :param random_state: (np.random.RandomState): random state object
        :return: samples from bootstrapped_events array of samples
        """
        random_instance = check_random_state(random_state)
        # Sequential Bootstrapping
        random_sample_index = random_instance.choice(range(len(bootstrapped_events)), p=prob)
        return bootstrapped_events[random_sample_index]

    def generate_unsampled_indices(self, bootstrapped_events, prob, n_samples, random_state):
        """Private function used to forest._set_oob_score function."""
        sample_indices = self.generate_sample_indices(bootstrapped_events, prob, random_state)
        sample_counts = np.bincount(sample_indices, minlength=n_samples)
        unsampled_mask = sample_counts == 0
        indices_range = np.arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]

        return unsampled_indices

    def parallel_build_trees(self, tree, forest, X, y, bootstrapped_events, prob, sample_weight, tree_idx, n_trees,
                             verbose=0, class_weight=None):
        """
        parallel_build_trees from sklearn
        (https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py)
        """
        if verbose > 1:
            print("building tree %d of %d" % (tree_idx + 1, n_trees))

        if forest.bootstrap:
            n_samples = X.shape[0]
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
            else:
                curr_sample_weight = sample_weight.copy()

            # Choose samples from pregenerated bootstrapped samples
            indices = self.generate_sample_indices(bootstrapped_events, prob, tree.random_state)
            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

            if class_weight == 'subsample':
                with catch_warnings():
                    simplefilter('ignore', DeprecationWarning)
                    curr_sample_weight *= compute_sample_weight('auto', y, indices)
            elif class_weight == 'balanced_subsample':
                curr_sample_weight *= compute_sample_weight('balanced', y, indices)

            tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
        else:
            tree.fit(X, y, sample_weight=sample_weight, check_input=False)

        return tree

    def fit(self, X, y, bootstrapped_events, prob, sample_weight=None):
        """
        This is a copy of fit function from sklearn. The only difference is instead of using
        standard bootstrapping method Sequential Bootstrapping is used instead
        """
        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(self.parallel_build_trees)(
                    t, self, X, y, bootstrapped_events, prob, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y, bootstrapped_events, prob)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    @abstractmethod
    def _set_oob_score(self, X, y, bootstrapped_events, prob):
        """_set_oob_score function from sklearn copy"""

    def _validate_y_class_weight(self, y):
        """
        _validate_y_class_weight from sklearn copy
        """
        return y, None

    def _validate_X_predict(self, X):
        """
        _validate_X_predict from sklearn copy()
        """
        check_is_fitted(self, 'estimators_')

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    @property
    def feature_importances_(self):
        """
        feature_importances_ from sklearn copy
        """
        check_is_fitted(self, 'estimators_')

        all_importances = Parallel(n_jobs=self.n_jobs,
                                   **_joblib_parallel_args(prefer='threads'))(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)


class SequentialForestClassifier(SequentialBaseForest, ClassifierMixin, metaclass=ABCMeta):
    """Base class for forest of trees-based classifiers which use Sequential Bootstrapping
    The most part of code is copied from ForestClassifier from sklearn
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

    def _set_oob_score(self, X, y, bootstrapped_events, prob):
        """
        _set_oob_score function copy from sklearn
        """
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = [np.zeros((n_samples, n_classes_[k]))
                       for k in range(self.n_outputs_)]

        for estimator in self.estimators_:
            unsampled_indices = super().generate_unsampled_indices(bootstrapped_events, prob, n_samples,
                                                                   estimator.random_state)
            p_estimator = estimator.predict_proba(X[unsampled_indices, :],
                                                  check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = [p_estimator]

            for k in range(self.n_outputs_):
                predictions[k][unsampled_indices, :] += p_estimator[k]

        for k in range(self.n_outputs_):
            if (predictions[k].sum(axis=1) == 0).any():
                warn("Some inputs do not have OOB scores. "
                     "This probably means too few trees were used "
                     "to compute any reliable oob estimates.")

            decision = (predictions[k] /
                        predictions[k].sum(axis=1)[:, np.newaxis])
            oob_decision_function.append(decision)
            oob_score += np.mean(y[:, k] ==
                                 np.argmax(predictions[k], axis=1), axis=0)

        if self.n_outputs_ == 1:
            self.oob_decision_function_ = oob_decision_function[0]
        else:
            self.oob_decision_function_ = oob_decision_function

        self.oob_score_ = oob_score / self.n_outputs_

    def _validate_y_class_weight(self, y):
        """
        _validate_y_class_weight function copy from sklearn
        """
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ('balanced', 'balanced_subsample')
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include '
                                     '"balanced" and "balanced_subsample". Given "%s".'
                                     % self.class_weight)
                if self.warm_start:
                    warn('class_weight presets "balanced" or "balanced_subsample" are '
                         'not recommended for warm_start if the fitted data '
                         'differs from the full dataset. In order to use '
                         '"balanced" weights, use compute_class_weight("balanced", '
                         'classes, y). In place of y you can use a large '
                         'enough sample of the full training set target to '
                         'properly estimate the class frequency '
                         'distributions. Pass the resulting weights as the '
                         'class_weight parameter.')

            if (self.class_weight != 'balanced_subsample' or
                    not self.bootstrap):
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight,
                                                              y_original)

        return y, expanded_class_weight

    def predict(self, X):
        """
        predict function copy from sklearn
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

            return predictions

    def predict_proba(self, X):
        """
        predict_proba method copy from sklearn
        """
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba,
                                            lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """
        predict_log_proba method copy from sklearn
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class SequentialForestRegressor(SequentialBaseForest, RegressorMixin, metaclass=ABCMeta):
    """
    Base class for forest of trees-based regressors which use Sequential Bootstrapping
    The most part of code is copied from ForestClassifier from sklearn
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

    def predict(self, X):
        """
        predict method copy from sklearn
        """
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat

    def _set_oob_score(self, X, y, bootstrapped_events, prob):
        """
        _set_oob_score from sklearn
        """
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            unsampled_indices = super().generate_unsampled_indices(bootstrapped_events, prob, n_samples,
                                                                   estimator.random_state)
            p_estimator = estimator.predict(
                X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few trees were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = \
                self.oob_prediction_.reshape((n_samples,))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k],
                                        predictions[:, k])

        self.oob_score_ /= self.n_outputs_


class SequentialBootstrappingRandomForestClassifier(SequentialForestClassifier):
    """
    RandomForestClassifier from sklearn with Sequential Bootstrapping instead of standard random bootstrapping
    """

    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class SequentialBootstrappingRandomForestRegressor(SequentialForestRegressor):
    """
    RandomForestRegressor from sklearn with Sequential Bootstrapping instead of standard random bootstrapping
    """

    def __init__(self,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class SequentialBootstrappingExtraTreesClassifier(SequentialForestClassifier):
    """
    ExtraTreesClassifier from sklearn with Sequential Bootstrapping instead of standard random bootstrapping
    """

    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class SequentialBootstrappingExtraTreesRegressor(SequentialForestRegressor):
    """
    ExtraTreesRegressor from sklearn with Sequential Bootstrapping instead of standard random bootstrapping
    """

    def __init__(self,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
