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
from mlfinlab.data_structures.base_bars import BaseBars


class ImbalanceBars(BaseBars):
    """
    Contains all of the logic to construct the imbalance bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_imbalance_bars which will create an instance of this
    class and then construct the imbalance bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, file_path, metric, num_prev_bars=3, exp_num_ticks_init=100000, batch_size=2e7,
                 analyse_thresholds=False):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: "dollar_imbalance"
        :param num_prev_bars: (Int) Window size for E[T]s
        :param exp_num_ticks_init: (Int) Initial number of expected ticks
        :param batch_size: (Int) Number of rows to read in from the csv, per batch
        :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
        """
        BaseBars.__init__(self, file_path, metric, batch_size)

        # Information bar properties
        self.num_prev_bars = num_prev_bars
        self.exp_num_ticks_init = exp_num_ticks_init
        # Expected number of ticks extracted from prev bars
        self.exp_num_ticks = self.exp_num_ticks_init
        self.num_ticks_bar = []  # List of number of ticks from previous bars

        self.cum_theta = 0
        self.expected_imbalance = np.nan
        self.imbalance_array = []

        self.analyse_thresholds = analyse_thresholds
        self.bars_thresholds = []  # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        self.open_price = None
        self.high_price, self.low_price = np.inf, np.inf
        self.cum_ticks, self.cum_dollar_value, self.cum_volume, self.cum_theta = 0, 0, 0, 0

    def _extract_bars(self, data):
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (List) of bars built using the current batch.
        """

        # Iterate over rows
        list_bars = []
        for row in data.values:
            # Set variables
            date_time = row[0]
            price = np.float(row[1])
            volume = row[2]

            self.cum_ticks += 1

            if self.open_price is None:
                self.open_price = price

            self.cum_volume += volume

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(
                self.high_price, self.low_price, price)

            # Imbalance calculations
            signed_tick = self._apply_tick_rule(price)
            imbalance = self._get_imbalance(price, signed_tick, volume)
            self.imbalance_array.append(imbalance)
            self.cum_theta += imbalance

            # Get expected imbalance for the first time, when num_ticks_init passed
            if not list_bars and np.isnan(self.expected_imbalance):
                self.expected_imbalance = self._get_expected_imbalance(
                    self.exp_num_ticks)

            if self.analyse_thresholds is True:
                self.bars_thresholds.append(
                    {'timestamp': date_time, 'cum_theta': self.cum_theta, 'exp_num_ticks': self.exp_num_ticks,
                     'exp_imbalance': np.abs(self.expected_imbalance)})

            self.prev_price = price  # Update previous price used for tick rule calculations

            # Check expression for possible bar generation
            if np.abs(self.cum_theta) > self.exp_num_ticks * np.abs(self.expected_imbalance):
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                self.num_ticks_bar.append(self.cum_ticks)
                # Expected number of ticks based on formed bars
                self.exp_num_ticks = ewma(np.array(
                    self.num_ticks_bar[-self.num_prev_bars:], dtype=float), self.num_prev_bars)[-1]

                self.expected_imbalance = self._get_expected_imbalance(
                    self.exp_num_ticks * self.num_prev_bars)
                # Reset counters
                self._reset_cache()

        return list_bars

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: Updated cum_ticks, cum_volume, cum_theta, high_price, low_price, exp_num_ticks.
        """
        # Check flag
        if self.flag and self.cache:
            latest_entry = self.cache[-1]

            # Update variables based on cache
            cum_ticks = int(latest_entry.cum_ticks)
            cum_volume = int(latest_entry.cum_volume)
            low_price = np.float(latest_entry.low)
            high_price = np.float(latest_entry.high)
            # cumulative imbalance for a particular imbalance calculation (theta_t in Prado book)
            cum_theta = np.float(latest_entry.cum_theta)
        else:
            # Reset counters
            cum_ticks, cum_theta, cum_volume = 0, 0, 0
            high_price, low_price = -np.inf, np.inf

        return cum_ticks, cum_volume, cum_theta, high_price, low_price

    def _update_cache(self, date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_theta):
        """
        Update the cache which is used to create a continuous flow of bars from one batch to the next.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param low_price: Lowest price in the period
        :param high_price: Highest price in the period
        :param cum_ticks: Cumulative number of ticks
        :param cum_volume: Cumulative volume (# of contracts)
        :param cum_theta: Cumulative Theta sub t (pg 29)
        """
        cache_data = self.cache_tuple(date_time=date_time, price=price, high=high_price, low=low_price,
                                      cum_ticks=cum_ticks, cum_volume=cum_volume, cum_theta=cum_theta)
        self.cache.append(cache_data)

    def _get_expected_imbalance(self, window):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
        :param window: EWMA window for calculation
        :return: expected_imbalance: 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(self.imbalance_array) < self.exp_num_ticks_init:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(min(len(self.imbalance_array), window))

        if np.isnan(ewma_window):
            # return nan, wait until len(self.imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(self.imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]

        return expected_imbalance


def get_dollar_imbalance_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                              batch_size=2e7, analyse_thresholds=False, verbose=True, to_csv=False, output_path=None):
    """
    Creates the dollar imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of dollar bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='dollar_imbalance', num_prev_bars=num_prev_bars,
                         exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size,
                         analyse_thresholds=analyse_thresholds)
    dollar_imbalance_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return dollar_imbalance_bars, bars.bars_thresholds


def get_volume_imbalance_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                              batch_size=2e7, verbose=True, to_csv=False, analyse_thresholds=False, output_path=None):
    """
    Creates the volume imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of volume bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='volume_imbalance', num_prev_bars=num_prev_bars,
                         exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size,
                         analyse_thresholds=analyse_thresholds)
    volume_imbalance_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_imbalance_bars, bars.bars_thresholds


def get_tick_imbalance_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                            batch_size=2e7, verbose=True, to_csv=False, analyse_thresholds=False, output_path=None):
    """
    Creates the tick imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of tick bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='tick_imbalance', num_prev_bars=num_prev_bars,
                         exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size,
                         analyse_thresholds=analyse_thresholds)
    tick_imbalance_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return tick_imbalance_bars
