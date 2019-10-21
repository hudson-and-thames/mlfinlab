"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures: Imbalance Bars

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of tick, volume, and dollar imbalance bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 29) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time
interval sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high
frequency paradigm, Lopez de Prado, et al. These ideas are then extended in another paper: Flow toxicity and liquidity
in a high-frequency world.

A good blog post to read, which helped us a lot in the implementation here is writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

# Imports
from collections import namedtuple
import numpy as np

from mlfinlab.util.fast_ewma import ewma
from mlfinlab.data_structures.base_bars import BaseBars, BaseImbalanceBars


class EMAImbalanceBars(BaseImbalanceBars):
    """
    Contains all of the logic to construct the imbalance bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_imbalance_bars which will create an instance of this
    class and then construct the imbalance bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, file_path, metric, expected_num_ticks_window, expected_imbalance_window, exp_num_ticks_init,
                 min_exp_num_ticks,
                 max_exp_num_ticks, batch_size,
                 analyse_thresholds):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: "dollar_imbalance"
        :param expected_num_ticks_window: (Int) Window size for E[T]s
        :param expected_imbalance_window: (Int) EMA window used to estimate expected imbalance
        :param exp_num_ticks_init: (Int) Initial number of expected ticks
        :param min_exp_num_ticks: (Int) Minimum number of expected ticks. Used to control bars sampling convergence
        :param max_exp_num_ticks: (Int) Maximum number of expected ticks. Used to control bars sampling convergence
        :param batch_size: (Int) Number of rows to read in from the csv, per batch
        :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
        """
        BaseImbalanceBars.__init__(file_path, metric, batch_size, expected_imbalance_window, expected_num_ticks_init, analyse_thresholds)

        # EMA Imbalance specific  hyper parameters
        self.expected_num_ticks_window = expected_num_ticks_window
        self.expected_imbalance_window = expected_imbalance_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.min_exp_num_ticks = min_exp_num_ticks
        self.max_exp_num_ticks = max_exp_num_ticks


    def _get_exp_num_ticks(self):
        exp_num_ticks = ewma(np.array(
            self.num_ticks_bar[-self.expected_num_ticks_window:], dtype=float), self.expected_num_ticks_window)[-1]
        return min(max(exp_num_ticks, self.min_exp_num_ticks), self.max_exp_num_ticks)

class ConstImbalanceBars(BaseImbalanceBars):
    """
    Contains all of the logic to construct the imbalance bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_imbalance_bars which will create an instance of this
    class and then construct the imbalance bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, file_path, metric, expected_imbalance_window, exp_num_ticks_init, batch_size,
                 analyse_thresholds):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: "dollar_imbalance"
        :param expected_num_ticks_window: (Int) Window size for E[T]s
        :param expected_imbalance_window: (Int) EMA window used to estimate expected imbalance
        :param exp_num_ticks_init: (Int) Initial number of expected ticks
        :param min_exp_num_ticks: (Int) Minimum number of expected ticks. Used to control bars sampling convergence
        :param max_exp_num_ticks: (Int) Maximum number of expected ticks. Used to control bars sampling convergence
        :param batch_size: (Int) Number of rows to read in from the csv, per batch
        :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
        """
        BaseImbalanceBars.__init__(file_path, metric, batch_size, expected_imbalance_window, expected_num_ticks_init, analyse_thresholds)

    def _get_exp_num_ticks(self):
        return self.exp_num_ticks


def get_ema_dollar_imbalance_bars(file_path, expected_num_ticks_window, exp_num_ticks_init=20000,
                              batch_size=2e7, analyse_thresholds=False, verbose=True, to_csv=False, output_path=None):
    """
    Creates the dollar imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param expected_num_ticks_window: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of dollar bars
    """
    bars = EMAImbalanceBars(file_path=file_path, metric='dollar_imbalance', expected_num_ticks_window=expected_num_ticks_window,
                         exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size,
                         analyse_thresholds=analyse_thresholds)
    dollar_imbalance_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return dollar_imbalance_bars, bars.bars_thresholds


def get_ema_volume_imbalance_bars(file_path, expected_num_ticks_window, exp_num_ticks_init=20000,
                              batch_size=2e7, verbose=True, to_csv=False, analyse_thresholds=False, output_path=None):
    """
    Creates the volume imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param expected_num_ticks_window: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of volume bars
    """
    bars = EMAImbalanceBars(file_path=file_path, metric='volume_imbalance', expected_num_ticks_window=expected_num_ticks_window,
                         exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size,
                         analyse_thresholds=analyse_thresholds)
    volume_imbalance_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_imbalance_bars, bars.bars_thresholds


def get_ema_tick_imbalance_bars(file_path, expected_num_ticks_window, exp_num_ticks_init=20000,
                            batch_size=2e7, verbose=True, to_csv=False, analyse_thresholds=False, output_path=None):
    """
    Creates the tick imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param expected_num_ticks_window: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of tick bars
    """
    bars = EMAImbalanceBars(file_path=file_path, metric='tick_imbalance', expected_num_ticks_window=expected_num_ticks_window,
                         exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size,
                         analyse_thresholds=analyse_thresholds)
    tick_imbalance_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return tick_imbalance_bars, bars.bars_thresholds
