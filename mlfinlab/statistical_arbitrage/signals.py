"""
Implements Signals.
"""

import numpy as np
import pandas as pd
import warnings


def z_score(data):
    """
    Calculates the z-score for the given data.

    :param data: (np.array) Data for z-score calculation.
    :return: (np.array) Z-score of the given data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.nan_to_num((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    return res

def ornstein_uhlenbeck():
    return
