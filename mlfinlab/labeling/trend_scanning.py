"""
Implementation of Trend-Scanning labels described in "Advances in Financial Machine Learning: Lecture 3/10"
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419
"""

import pandas as pd
import numpy as np
from mlfinlab.structural_breaks.sadf import _get_betas


def trend_scanning_labels(price_series: pd.Series, t_events: list = None, look_forward_window: int = 20,
                          min_sample_length: int = 5, step: int = 1) -> pd.DataFrame:
    """
    Get Trend-Scanning labels (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419)

    :param price_series: (pd.Series) close prices used to label the data set
    :param t_events: (list) of filtered events, array of pd.Timestamps
    :param look_forward_window: (int) maximum look forward window used to get the trend value
    :param min_sample_length: (int) minimum sample length used to fit regression
    :param step: (int) optimal t-value index is searched every 'step' indices
    :return: (pd.DataFrame) of t1, ret, bin (label information). t1 - label endtime, ret - price change %,
                            bin - label value based on price change sign
    """
    # pylint: disable=invalid-name

    if t_events is None:
        t_events = price_series.index

    t1_array = []  # Array of label end times
    t_values_array = []  # Array of trend t-values

    for index in t_events:
        subset = price_series.loc[index:].iloc[:look_forward_window]  # Take t:t+L window
        if subset.shape[0] >= look_forward_window:
            # Loop over possible look-ahead windows to get the one which yields maximum t values for b_1 regression coef
            max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
            max_t_value_index = None  # Index with maximum t-value
            max_t_value = None  # Maximum t-value signed

            # Get optimal label endtime value based on regression t-statistics
            for forward_window in np.arange(min_sample_length, subset.shape[0], step):
                y_subset = subset.iloc[:forward_window].values.reshape(-1, 1)  # y{t}:y_{t+l}

                # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
                X_subset = np.ones((y_subset.shape[0], 2))
                X_subset[:, 1] = np.arange(y_subset.shape[0])

                # Get regression coefficients estimates
                b_mean_, b_std_ = _get_betas(X_subset, y_subset)
                # Check if l gives the maximum t-value among all values {0...L}
                t_beta_1 = (b_mean_[1] / np.sqrt(b_std_[1, 1]))[0]
                if abs(t_beta_1) > max_abs_t_value:
                    max_abs_t_value = abs(t_beta_1)
                    max_t_value = t_beta_1
                    max_t_value_index = forward_window

            # Store label information (t1, return)
            label_endtime_index = subset.index[max_t_value_index - 1]
            t1_array.append(label_endtime_index)
            t_values_array.append(max_t_value)

        else:
            t1_array.append(None)
            t_values_array.append(None)

    labels = pd.DataFrame({'t1': t1_array, 't_value': t_values_array}, index=t_events)
    labels.loc[:, 'ret'] = price_series.loc[labels.t1].values / price_series.loc[labels.index].values - 1
    labels['bin'] = np.sign(labels.t_value)
    return labels
