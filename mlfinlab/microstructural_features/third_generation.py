"""
Third generation models implementation (VPIN)
"""
import pandas as pd


def get_vpin(volume: pd.Series, buy_volume: pd.Series, window: int = 1) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 292-293.

    Get Volume-Synchronized Probability of Informed Trading (VPIN) from bars

    :param volume: (pd.Series) Bar volume
    :param buy_volume: (pd.Series) Bar volume classified as buy (either tick rule, BVC or aggressor side methods applied)
    :param window: (int) Estimation window
    :return: (pd.Series) VPIN series
    """
    sell_volume = volume - buy_volume
    volume_imbalance = abs(buy_volume - sell_volume)
    return volume_imbalance.rolling(window=window).mean() / volume
