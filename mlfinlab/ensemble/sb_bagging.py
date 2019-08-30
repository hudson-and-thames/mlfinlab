"""
Implementation of Sequentially Bootstrapped Bagging Classifier using sklearn's library as base class
"""
import numbers
import itertools
from warnings import warn
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble.bagging import BaseBagging, BaggingClassifier, BaggingRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble.base import _partition_estimators
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import indices_to_mask
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils import check_random_state, check_array, check_consistent_length, check_X_y
from sklearn.utils._joblib import Parallel, delayed

from mlfinlab.sampling.bootstrapping import seq_bootstrap, get_ind_matrix

MAX_INT = np.iinfo(np.int32).max


# pylint: disable=too-many-ancestors
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=len-as-condition
# pylint: disable=attribute-defined-outside-init
# pylint: disable=bad-super-call
# pylint: disable=no-else-raise


def _generate_random_features(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indices."""
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)

    return indices


def _generate_bagging_indices(random_state, bootstrap_features, n_features, max_features, max_samples, ind_mat):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_random_features(random_state, bootstrap_features,
                                                n_features, max_features)
    sample_indices = seq_bootstrap(ind_mat, sample_length=max_samples, random_state=random_state)

    return feature_indices, sample_indices


def _parallel_build_estimators(n_estimators, ensemble, X, y, ind_mat, sample_weight,
                               seeds, total_n_estimators, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")

    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []
    estimators_indices = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      n_features,
                                                      max_features,
                                                      max_samples,
                                                      ind_mat)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)
        estimators_indices.append(indices)

    return estimators, estimators_features, estimators_indices


class SequentiallyBootstrappedBaseBagging(BaseBagging, metaclass=ABCMeta):
    """
    Base class for Sequentially Bootstrapped Classifier and Regressor, extension of sklearn's BaseBagging
    """

    @abstractmethod
    def __init__(self,
                 samples_info_sets,
                 price_bars,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=True,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        # pylint: disable=invalid-name
        self.samples_info_sets = samples_info_sets
        self.price_bars = price_bars
        self.ind_mat = get_ind_matrix(samples_info_sets, price_bars)
        # Used for create get ind_matrix subsample during cross-validation
        self.timestamp_int_index_mapping = pd.Series(index=samples_info_sets.index,
                                                     data=range(self.ind_mat.shape[1]))

        self.X_time_index = None  # Timestamp index of X_train

    def fit(self, X, y, sample_weight=None):
        """Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
        """
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.
        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
        """
        random_state = check_random_state(self.random_state)
        self.X_time_index = X.index  # Remember X index for future sampling

        # Generate subsample ind_matrix (we need this during subsampling cross_validation)
        subsampled_ind_mat = self.ind_mat[:, self.timestamp_int_index_mapping.loc[self.X_time_index]]

        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        # Validate max_samples
        if not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []
            self.sequentially_bootstrapped_samples_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        # pylint: disable=C0330
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                               )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                subsampled_ind_mat,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        self.sequentially_bootstrapped_samples_ += list(itertools.chain.from_iterable(
            t[2] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self


class SequentiallyBootstrappedBaggingClassifier(SequentiallyBootstrappedBaseBagging, BaggingClassifier,
                                                ClassifierMixin):
    """
    A Sequentially Bootstrapped Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset generated using
    Sequential Bootstrapping sampling procedure and then aggregate their individual predictions (
    either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    :param samples_info_sets: pd.Series, The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param price_bars: pd.DataFrame
        Price bars used in samples_info_sets generation
    :param base_estimator: object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    :param n_estimators: int, optional (default=10)
        The number of base estimators in the ensemble.
    :param max_samples: int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw `max_samples` samples. If float, then draw `max_samples * X.shape[0]` samples.
    :param max_features: int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw `max_features` features. If float, then draw `max_features * X.shape[1]` features.
    :param bootstrap_features: boolean, optional (default=False)
        Whether features are drawn with replacement.
    :param oob_score: bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization error.
    :param warm_start: bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    :param n_jobs: int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :param random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    :ivar base_estimator_: estimator
        The base estimator from which the ensemble is grown.
    :ivar estimators_: list of estimators
        The collection of fitted base estimators.
    :ivar estimators_samples_: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    :ivar estimators_features_: list of arrays
        The subset of drawn features for each base estimator.
    :ivar classes_: array of shape = [n_classes]
        The classes labels.
    :ivar n_classes_: int or list
        The number of classes.
    :ivar oob_score_: float
        Score of the training dataset obtained using an out-of-bag estimate.
    :ivar oob_decision_function_: array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    """

    def __init__(self,
                 samples_info_sets,
                 price_bars,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            samples_info_sets=samples_info_sets,
            price_bars=price_bars,
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaggingClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.sequentially_bootstrapped_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        if (predictions.sum(axis=1) == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")

        oob_decision_function = (predictions /
                                 predictions.sum(axis=1)[:, np.newaxis])
        oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score


class SequentiallyBootstrappedBaggingRegressor(SequentiallyBootstrappedBaseBagging, BaggingRegressor, RegressorMixin):
    """
    A Sequentially Bootstrapped Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset using Sequential Bootstrapping and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    :param samples_info_sets: pd.Series, The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.

    :param price_bars: pd.DataFrame
        Price bars used in samples_info_sets generation
    :param base_estimator: object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.
    :param n_estimators: int, optional (default=10)
        The number of base estimators in the ensemble.
    :param max_samples: int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw `max_samples` samples. If float, then draw `max_samples * X.shape[0]` samples.
    :param max_features: int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw `max_features` features. If float, then draw `max_features * X.shape[1]` features.
    :param bootstrap_features: boolean, optional (default=False)
        Whether features are drawn with replacement.
    :param oob_score: bool
        Whether to use out-of-bag samples to estimate
        the generalization error.
    :param warm_start: bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    :param n_jobs: int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :param random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    :ivar estimators_: list of estimators
        The collection of fitted sub-estimators.
    :ivar estimators_samples_: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    :ivar estimators_features_: list of arrays
        The subset of drawn features for each base estimator.
    :ivar oob_score_: float
        Score of the training dataset obtained using an out-of-bag estimate.
    :ivar oob_prediction_: array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_prediction_` might contain NaN.
    """

    def __init__(self,
                 samples_info_sets,
                 price_bars,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            samples_info_sets=samples_info_sets,
            price_bars=price_bars,
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaggingRegressor, self)._validate_estimator(
            default=DecisionTreeRegressor())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(self.estimators_,
                                                self.sequentially_bootstrapped_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            predictions[mask] += estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)
