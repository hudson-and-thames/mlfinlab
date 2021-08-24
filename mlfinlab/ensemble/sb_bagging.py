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
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble._bagging import BaseBagging
from sklearn.ensemble._base import _partition_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
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

    pass


def _generate_bagging_indices(random_state, bootstrap_features, n_features, max_features, max_samples, ind_mat):
    """Randomly draw feature and sample indices."""

    pass

def _parallel_build_estimators(n_estimators, ensemble, X, y, ind_mat, sample_weight,
                               seeds, total_n_estimators, verbose):
    """Private function used to build a batch of estimators within a job."""

    pass


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


        pass

    def fit(self, X, y, sample_weight=None):
        """Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : (array-like, sparse matrix) of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : (array-like), shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : (array-like), shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : (object)
        """

        pass

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : (array-like, sparse matrix) of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : (array-like), shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : (int or float), optional (default=None)
            Argument to use instead of self.max_samples.
        max_depth : (int), optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : (array-like), shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : (object)
        """

        pass


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

    :param samples_info_sets: (pd.Series), The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param price_bars: (pd.DataFrame)
        Price bars used in samples_info_sets generation
    :param base_estimator: (object or None), optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    :param n_estimators: (int), optional (default=10)
        The number of base estimators in the ensemble.
    :param max_samples: (int or float), optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw `max_samples` samples. If float, then draw `max_samples * X.shape[0]` samples.
    :param max_features: (int or float), optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw `max_features` features. If float, then draw `max_features * X.shape[1]` features.
    :param bootstrap_features: (bool), optional (default=False)
        Whether features are drawn with replacement.
    :param oob_score: (bool), optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization error.
    :param warm_start: (bool), optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    :param n_jobs: (int or None), optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :param random_state: (int), RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: (int), optional (default=0)
        Controls the verbosity when fitting and predicting.

    :ivar base_estimator_: (estimator)
        The base estimator from which the ensemble is grown.
    :ivar estimators_: (list of estimators)
        The collection of fitted base estimators.
    :ivar estimators_samples_: (list of arrays)
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    :ivar estimators_features_: (list of arrays)
        The subset of drawn features for each base estimator.
    :ivar classes_: (array) of shape = [n_classes]
        The classes labels.
    :ivar n_classes_: (int or list)
        The number of classes.
    :ivar oob_score_: (float)
        Score of the training dataset obtained using an out-of-bag estimate.
    :ivar oob_decision_function_: (array) of shape = [n_samples, n_classes]
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

        pass

    def _set_oob_score(self, X, y):

        pass


class SequentiallyBootstrappedBaggingRegressor(SequentiallyBootstrappedBaseBagging, BaggingRegressor, RegressorMixin):
    """
    A Sequentially Bootstrapped Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset using Sequential Bootstrapping and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    :param samples_info_sets: (pd.Series), The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.

    :param price_bars: (pd.DataFrame)
        Price bars used in samples_info_sets generation
    :param base_estimator: (object or None), optional (default=None)
        The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.
    :param n_estimators: (int), optional (default=10)
        The number of base estimators in the ensemble.
    :param max_samples: (int or float), optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw `max_samples` samples. If float, then draw `max_samples * X.shape[0]` samples.
    :param max_features: (int or float), optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw `max_features` features. If float, then draw `max_features * X.shape[1]` features.
    :param bootstrap_features: (bool), optional (default=False)
        Whether features are drawn with replacement.
    :param oob_score: (bool)
        Whether to use out-of-bag samples to estimate
        the generalization error.
    :param warm_start: (bool), optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    :param n_jobs: (int or None), optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :param random_state: (int, RandomState instance or None), optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: (int), optional (default=0)
        Controls the verbosity when fitting and predicting.

    :ivar estimators_: (list) of estimators
        The collection of fitted sub-estimators.
    :ivar estimators_samples_: (list) of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    :ivar estimators_features_: (list) of arrays
        The subset of drawn features for each base estimator.
    :ivar oob_score_: (float)
        Score of the training dataset obtained using an out-of-bag estimate.
    :ivar oob_prediction_: (array) of shape = [n_samples]
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

        pass

    def _set_oob_score(self, X, y):

        pass
