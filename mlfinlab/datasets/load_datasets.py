"""
The module implementing various functions loading tick, dollar, stock data sets which can be used as
sandbox data
"""

import os
import pandas as pd


def load_stock_prices() -> pd.DataFrame:
    """
    Loads stock prices data sets consisting of
    EEM, EWG, TIP, EWJ, EFA, IEF, EWQ, EWU, XLB, XLE, XLF, LQD, XLK, XLU, EPP, FXI, VGK, VPL, SPY, TLT, BND, CSJ,
    DIA starting from 2008 till 2016.

    :return: (pd.DataFrame) stock_prices data frame
    """

    pass


def load_tick_sample() -> pd.DataFrame:
    """
    Loads E-Mini S&P 500 futures tick data sample

    :return: (pd.DataFrame) with tick data sample
    """

    pass


def load_dollar_bar_sample() -> pd.DataFrame:
    """
    Loads E-Mini S&P 500 futures dollar bars data sample.

    :return: (pd.DataFrame) with dollar bar data sample
    """

    pass
