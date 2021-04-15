"""
Detection of bull and bear markets.
"""
import numpy as np
import pandas as pd


def pagan_sossounov(prices, window=8, censor=6, cycle=16, phase=4, threshold=0.2):
    """
    Pagan and Sossounov's labeling method. Sourced from `Pagan, Adrian R., and Kirill A. Sossounov. "A simple framework
    for analysing bull and bear markets." Journal of applied econometrics 18.1 (2003): 23-46.
    <https://onlinelibrary.wiley.com/doi/pdf/10.1002/jae.664>`__

    Returns a DataFrame with labels of 1 for Bull and -1 for Bear.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market.
    :param window: (int) Rolling window length to determine local extrema. Paper suggests 8 months for monthly obs.
    :param censor: (int) Number of months to eliminate for start and end. Paper suggests 6 months for monthly obs.
    :param cycle: (int) Minimum length for a complete cycle. Paper suggests 16 months for monthly obs.
    :param phase: (int) Minimum length for a phase. Paper suggests 4 months for monthly obs.
    :param threshold: (double) Minimum threshold for phase change. Paper suggests 0.2.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    pass


def _alternation(price):
    """
    Helper function to check peak and trough alternation.

    :param price: (pd.DataFrame) Close prices of all tickers in the market.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    pass


def _apply_pagan_sossounov(price, window, censor, cycle, phase, threshold):
    """
    Helper function for Pagan and Sossounov labeling method.

    :param price: (pd.DataFrame) Close prices of all tickers in the market.
    :param window: (int) Rolling window length to determine local extrema. Paper suggests 8 months for monthly obs.
    :param censor: (int) Number of months to eliminate for start and end. Paper suggests 6 months for monthly obs.
    :param cycle: (int) Minimum length for a complete cycle. Paper suggests 16 months for monthly obs.
    :param phase: (int) Minimum length for a phase. Paper suggests 4 months for monthly obs.
    :param threshold: (double) Minimum threshold for phase change. Paper suggests 20%.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    pass


def lunde_timmermann(prices, bull_threshold=0.15, bear_threshold=0.15):
    """
    Lunde and Timmermann's labeling method. Sourced from `Lunde, Asger, and Allan Timmermann. "Duration dependence
    in stock prices: An analysis of bull and bear markets." Journal of Business & Economic Statistics 22.3 (2004): 253-273.
    <https://repec.cepr.org/repec/cpr/ceprdp/DP4104.pdf>`__

    Returns a DataFrame with labels of 1 for Bull and -1 for Bear.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market.
    :param bull_threshold: (double) Threshold to identify bull market. Paper suggests 0.15.
    :param bear_threshold: (double) Threshold to identify bear market. Paper suggests 0.15.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    pass


def _apply_lunde_timmermann(price, bull_threshold, bear_threshold):
    """
    Helper function for Lunde and Timmermann labeling method.

    :param price: (pd.DataFrame) Close prices of all tickers in the market.
    :param bull_threshold: (double) Threshold to identify bull market. Paper suggests 0.15.
    :param bear_threshold: (double) Threshold to identify bear market. Paper suggests 0.15.
    :return: (pd.DataFrame) Labeled pd.DataFrame. 1 for Bull, -1 for Bear.
    """

    pass
