"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of tick, volume, and dollar run bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 31) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al. These ideas are then extended in another paper: Flow toxicity and liquidity
in a high-frequency world.

We have introduced two types of run bars: with expected number of tick defined through EMA (book implementation) and
constant number of ticks.

A good blog post to read, which helped us a lot in the implementation here is writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

# Imports
from typing import Union, Iterable, List, Optional

import numpy as np
import pandas as pd

from mlfinlab.data_structures.base_bars import BaseRunBars
from mlfinlab.util.fast_ewma import ewma


class EMARunBars(BaseRunBars):
    """
    Contains all of the logic to construct the run bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_ema_dollar_imbalance_bars which will create an instance of this
    class and then construct the run bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, num_prev_bars: int,
                 expected_imbalance_window: int, exp_num_ticks_init: int,
                 exp_num_ticks_constraints: List[float], batch_size: int, analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar_run"
        :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
        :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance
        :param exp_num_ticks_init: (int) Initial number of expected ticks
        :param exp_num_ticks_constraints (list) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        :param analyse_thresholds: (bool) Flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a
                                          form of Pandas DataFrame
        """

        pass

    def _get_exp_num_ticks(self):

        pass


class ConstRunBars(BaseRunBars):
    """
    Contains all of the logic to construct the imbalance bars with fixed expected number of ticks. This class shouldn't
    be used directly. We have added functions to the package such as get_const_dollar_imbalance_bars which will create
    an instance of this class and then construct the imbalance bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, num_prev_bars: int,
                 expected_imbalance_window: int, exp_num_ticks_init: int, batch_size: int,
                 analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar_run"
        :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
        :param expected_imbalance_window: (int) EMA window used to estimate expected run
        :param exp_num_ticks_init: (int) Initial number of expected ticks
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars
        """

        pass

    def _get_exp_num_ticks(self):

        pass


def get_ema_dollar_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int = 3,
                            expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                            exp_num_ticks_constraints: List[float] = None, batch_size: int = 2e7,
                            analyse_thresholds: bool = False,
                            verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the EMA dollar run bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
    :param expected_imbalance_window: (int) EMA window used to estimate expected run
    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
    :param exp_num_ticks_constraints: (list) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) DataFrame of dollar bars and DataFrame of thresholds
    """

    pass


def get_ema_volume_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int = 3,
                            expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                            exp_num_ticks_constraints: List[float] = None, batch_size: int = 2e7,
                            analyse_thresholds: bool = False,
                            verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the EMA volume run bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_pats_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
    :param expected_imbalance_window: (int) EMA window used to estimate expected run
    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
    :param exp_num_ticks_constraints: (list) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) DataFrame of volume bars and DataFrame of thresholds
    """

    pass


def get_ema_tick_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int = 3,
                          expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                          exp_num_ticks_constraints: List[float] = None, batch_size: int = 2e7,
                          analyse_thresholds: bool = False,
                          verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the EMA tick run bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)
    :param expected_imbalance_window: (int) EMA window used to estimate expected run
    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
    :param exp_num_ticks_constraints: (list) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) DataFrame of tick bars and DataFrame of thresholds
    """

    pass


def get_const_dollar_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int,
                              expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                              batch_size: int = 2e7, analyse_thresholds: bool = False,
                              verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the Const dollar run bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param num_prev_bars: (int) Window size for estimating buy ticks proportion (number of previous bars to use in EWMA)
    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance
    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) DataFrame of dollar bars and DataFrame of thresholds, if to_csv=True returns None
    """

    pass


def get_const_volume_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int,
                              expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                              batch_size: int = 2e7, analyse_thresholds: bool = False,
                              verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the Const volume run bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param num_prev_bars: (int) Window size for estimating buy ticks proportion (number of previous bars to use in EWMA)
    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance
    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) DataFrame of volume bars and DataFrame of thresholds
    """

    pass


def get_const_tick_run_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int,
                            expected_imbalance_window: int = 10000, exp_num_ticks_init: int = 20000,
                            batch_size: int = 2e7, analyse_thresholds: bool = False,
                            verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the Const tick run bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param num_prev_bars: (int) Window size for estimating buy ticks proportion (number of previous bars to use in EWMA)
    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance
    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample run bars
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) DataFrame of tick bars and DataFrame of thresholds
    """

    pass
