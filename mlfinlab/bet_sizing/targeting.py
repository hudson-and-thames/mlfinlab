"""
This module contains functionality for determining bet sizes for investments based on volatility targeting.
"""

import numpy as np


def volatility_targeting(returns: np.ndarray, target_vol: float, annualize_factor: int):
    """
    Calculate the position size to make the expected volatility of the portfolio be the target variability.

    :param returns: (np.ndarray) This is series of returns of your strategy or a asset.
    :param target_vol: (float) volatility(%) that you target.
    :return: (float) position size you should use to change the volatility to be target volatility.
    """
    annualized_volatility = np.std(returns, ddof=1) * np.sqrt(annualize_factor)
    position_size = (target_vol / 100.) / annualized_volatility

    return position_size
