from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing
from ..sampling.bootstrapping import get_ind_matrix, seq_bootstrap
from sklearn.utils.multiclass import check_classification_targets
from imblearn.utils import check_sampling_strategy
from sklearn.preprocessing import label_binarize
import numpy as np


class SequentialBootstrappingSampler(BaseUnderSampler):
    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 ratio=None):
        super().__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.sample_indices_ = None

    def fit_resample(self, X, y, **kwargs):
        check_classification_targets(y)
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)

        output = self._fit_resample(X, y, **kwargs)

        if binarize_y:
            y_sampled = label_binarize(output[1], np.unique(y))
            if len(output) == 2:
                return output[0], y_sampled
            return output[0], y_sampled, output[2]
        return output

    def _fit_resample(self, X, y, **kwargs):
        print(kwargs)
        triple_barrier_events = kwargs['triple_barrier_events']
        random_state = check_random_state(self.random_state)
        ind_matrix = get_ind_matrix(triple_barrier_events)
        num_samples_to_draw = None # this feature will be added in the future
        bootstrapped_samples_idx = seq_bootstrap(ind_matrix, sample_length=num_samples_to_draw,
                                                 )

        self.sample_indices_ = bootstrapped_samples_idx

        if self.return_indices:
            return (safe_indexing(X, bootstrapped_samples_idx), safe_indexing(y, bootstrapped_samples_idx),
                    bootstrapped_samples_idx)
        return safe_indexing(X, bootstrapped_samples_idx), safe_indexing(y, bootstrapped_samples_idx)
