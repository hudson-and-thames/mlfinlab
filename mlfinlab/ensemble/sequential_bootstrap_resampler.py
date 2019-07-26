from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing
from ..sampling.bootstrapping import get_ind_matrix, seq_bootstrap


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

    def _fit_resample(self, X, triple_barrier_events, dollar_bars):
        y = triple_barrier_events.bin
        random_state = check_random_state(self.random_state)
        ind_matrix = get_ind_matrix(triple_barrier_events)
        num_samples_to_draw = X.shape * self.ratio
        bootstrapped_samples_idx = seq_bootstrap(ind_matrix, sample_length=num_samples_to_draw,
                                                 random_state=random_state)

        self.sample_indices_ = bootstrapped_samples_idx

        if self.return_indices:
            return (safe_indexing(X, bootstrapped_samples_idx), safe_indexing(y, bootstrapped_samples_idx),
                    bootstrapped_samples_idx)
        return safe_indexing(X, bootstrapped_samples_idx), safe_indexing(y, bootstrapped_samples_idx)
