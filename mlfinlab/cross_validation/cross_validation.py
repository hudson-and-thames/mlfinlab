"""
Implements the book chapter 7 on Cross Validation for financial data
"""

import pandas as pd


def get_train_times(observations: pd.Series, test_times: pd.Series) -> pd.Series:  # pragma: no cover
    """
    Given test_times, find the times of the training observations.
    —observations.index: Time when the observation started.
    —observations.value: Time when the observation ended.
    —test_times: Times of testing observations.
    """
    train = observations.copy(deep=True)
    for start_ix, end_ix in test_times.iteritems():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index  # train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index  # train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index  # train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train
