"""
Implementation of Trend-Scanning labels described in "Advances in Financial Machine Learning: Lecture 3/10"
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419
"""

import pandas as pd
import numpy as np
from mlfinlab.structural_breaks.sadf import _get_betas


def get_trend_scanning_labels(price_series: pd.Series, t_events: list, look_forward_window: int) -> pd.DataFrame:
    """
    Get Trend-Scanning labels (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419)

    :param price_series: (pd.Series) close prices used to label the data set
    :param t_events: (list) of filtered events, array of pd.Timestamps
    :param look_forward_window: (int) maximum look forward window used to get the trend value
    :return: (pd.DataFrame) of t1, ret, bin (label information). t1 - label endtime, ret - price change %,
                            bin - label value based on price change sign
    """
    # pylint: disable=invalid-name

    t1_array = []  # Array of label end times
    ret_array = []  # Array of returns

    for index in t_events:
        subset = price_series.loc[index:].iloc[:look_forward_window]  # Take t:t+L window
        if subset.shape[0] >= look_forward_window:
            # Loop over possible look-ahead windows to get the one which yields maximum t values for b_1 regression coef
            max_t_value = -np.inf  # Maximum t-value of b_1 coefficient among l values
            max_t_value_l = None  # Index with maximum t-value

            # Get optimal label endtime value based on regression t-statistics
            for forward_window in np.arange(subset.shape[0]):
                y_subset = subset.iloc[:forward_window + 1].values.reshape(-1, 1)  # y{t}:y_{t+l}

                # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
                X_subset = np.concatenate(
                    (np.ones(shape=(y_subset.shape[0], 1)), np.arange(forward_window + 1).reshape(-1, 1)),
                    axis=1)

                # Get regression coefficients estimates
                b_mean_, b_std_ = _get_betas(X_subset, y_subset)
                # Check if l gives the maximum t-value among all values {0...L}
                if len(b_mean_) > 1:
                    t_beta_1 = abs(b_mean_[1] / b_std_[1, 1])[0]
                    if t_beta_1 > max_t_value:
                        max_t_value = t_beta_1
                        max_t_value_l = forward_window

            # Store label information (t1, return)
            label_endtime_index = subset.index[max_t_value_l]
            price_pct_change = subset.loc[label_endtime_index] / subset.iloc[0] - 1
            t1_array.append(label_endtime_index)
            ret_array.append(float(price_pct_change))

        else:
            t1_array.append(None)
            ret_array.append(None)

    labels = pd.DataFrame({'t1': t1_array, 'ret': ret_array}, index=t_events)
    labels['bin'] = np.sign(labels.ret)
    return labels
