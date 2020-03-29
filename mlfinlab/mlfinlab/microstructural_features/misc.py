"""
Various miscellaneous microstructural features (VWAP, average tick size)
"""

import numpy as np


def vwap(dollar_volume: list, volume: list) -> float:
    """
    Get Volume Weighted Average Price (VWAP).

    :param dollar_volume: (list) of dollar volumes
    :param volume: (list) of trades sizes
    :return: (float) VWAP value
    """
    return sum(dollar_volume) / sum(volume)


def get_avg_tick_size(tick_size_arr: list) -> float:
    """
    Get average tick size in a bar.
    :param tick_size_arr: (list) of trade sizes
    :return: (float) average trade size
    """
    return np.mean(tick_size_arr)
