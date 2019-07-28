
import numbers

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.pipeline import Pipeline
from .sequential_bootstrap_resampler import SequentialBootstrappingSampler
from imblearn.utils import check_target_type


class SequentialBootstrappingBaggingClassifier(BaggingClassifier):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 sampling_strategy='auto',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 ratio=None):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.sampling_strategy = sampling_strategy
        self.ratio = ratio

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {}.".format(self.n_estimators))

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = clone(default)

        self.base_estimator_ = Pipeline([('sampler', SequentialBootstrappingSampler(
            sampling_strategy=self.sampling_strategy,
            ratio=self.ratio)), ('classifier', base_estimator)])

    def fit(self, X, y, triple_barrier_events):
        check_target_type(y)
        # RandomUnderSampler is not supporting sample_weight. We need to pass
        # None.
        return self._fit(X, y, self.max_samples, triple_barrier_events=triple_barrier_events)
