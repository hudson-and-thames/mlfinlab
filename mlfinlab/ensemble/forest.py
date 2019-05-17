import numpy as np
from scipy.sparse import issparse
from warnings import catch_warnings, simplefilter, warn
from sklearn.ensemble.forest import BaseForest, ForestClassifier, ForestRegressor
from sklearn.utils.fixes import parallel_helper, _joblib_parallel_args
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.exceptions import DataConversionWarning
from sklearn.utils._joblib import Parallel, delayed
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                    ExtraTreeClassifier, ExtraTreeRegressor)
from sklearn.metrics import r2_score
from mlfinlab.sampling.bootstrapping import seq_bootstrap

MAX_INT = np.iinfo(np.int32).max


class SequentialBaseForest(BaseForest):

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
        super().__init__(self,
                         base_estimator,
                         n_estimators,
                         estimator_params,
                         bootstrap,
                         oob_score,
                         n_jobs,
                         random_state,
                         verbose,
                         warm_start,
                         class_weight
                         )

    def apply(self, X):
        super().apply(X)

    def decision_path(self, X):
        super().decision_path(X)

    def generate_sample_indices(self, triple_barrier_events, random_state, n_samples):
        """Private function used to _parallel_build_trees function."""
        random_instance = check_random_state(random_state)
        # Sequential Bootstrapping
        sample_indices = seq_bootstrap(triple_barrier_events, random_state=random_instance, sample_length=n_samples)
        return sample_indices

    def generate_unsampled_indices(self, triple_barrier_events, random_state, n_samples):
        """Private function used to forest._set_oob_score function."""
        sample_indices = self.generate_sample_indices(triple_barrier_events, random_state, n_samples)
        sample_counts = np.bincount(sample_indices, minlength=n_samples)
        unsampled_mask = sample_counts == 0
        indices_range = np.arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]

        return unsampled_indices

    def parallel_build_trees(self, tree, forest, X, y, triple_barrier_events, sample_weight, tree_idx, n_trees,
                             verbose=0, class_weight=None):
        """Private function used to fit a single tree in parallel."""
        if verbose > 1:
            print("building tree %d of %d" % (tree_idx + 1, n_trees))

        if forest.bootstrap:
            n_samples = X.shape[0]
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
            else:
                curr_sample_weight = sample_weight.copy()

            # Sequential Bootstrapping part
            indices = self.generate_sample_indices(triple_barrier_events, tree.random_state, n_samples)
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

    def fit(self, X, y, triple_barrier_events, sample_weight=None):
        """
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
                    t, self, X, y, triple_barrier_events, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y, triple_barrier_events)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def _set_oob_score(self, X, y, triple_barrier_events):
        """Calculate out of bag predictions and score."""

    def _validate_y_class_weight(self, y):
        super()._validate_y_class_weight(y)

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        super()._validate_X_predict(X)

    def feature_importances_(self):
        """
        """
        super().feature_importances


class SequentialForestClassifier(ForestClassifier, SequentialBaseForest):
    """Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

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
        super().__init__(self,
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

    def _set_oob_score(self, X, y, triple_barrier_events):
        """Compute out-of-bag score"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_classes_ = self.n_classes_
        n_samples = y.shape[0]

        oob_decision_function = []
        oob_score = 0.0
        predictions = [np.zeros((n_samples, n_classes_[k]))
                       for k in range(self.n_outputs_)]

        for estimator in self.estimators_:
            unsampled_indices = super().generate_unsampled_indices(triple_barrier_events, estimator.random_state,
                                                                   n_samples)
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
        super()._validate_y_class_weight(y)

    def predict(self, X):
        super().predict(X)

    def predict_proba(self, X):
        super().predict_proba(X)

    def predict_log_proba(self, X):
        super().predict_log_proba(X)


class SequentialForestRegressor(ForestRegressor, SequentialBaseForest):
    """
    """

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
        super().predict(X)

    def _set_oob_score(self, X, y, triple_barrier_events):
        """Compute out-of-bag scores"""
        X = check_array(X, dtype=DTYPE, accept_sparse='csr')

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        for estimator in self.estimators_:
            unsampled_indices = super().generate_unsampled_indices(triple_barrier_events, estimator.random_state,
                                                                   n_samples)
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
