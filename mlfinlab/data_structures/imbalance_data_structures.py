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

    def __init__(self, file_path, metric, num_prev_bars=3, imbalance_ewma_window=None, exp_num_ticks_init=100000,
                 batch_size=2e7):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: "dollar_imbalance"
        :param num_prev_bars: (Int) Window size for E[T]s
        :param imbalance_ewma_window: (Int) Window size for imblance calculation
        :param exp_num_ticks_init: (Int) Initial number of expected ticks
        :param batch_size: (Int) Number of rows to read in from the csv, per batch
        """
        BaseBars.__init__(self, file_path, metric, batch_size)

        # Information bar properties
        self.num_prev_bars = num_prev_bars
        self.imbalance_ewma_window = imbalance_ewma_window
        self.exp_num_ticks_init = exp_num_ticks_init
        self.num_ticks_bar = []  # List of number of ticks from previous bars

        # Named tuple to help with storing the cache
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'cum_volume', 'cum_theta', 'exp_num_ticks',
                                       'imbalance_array'])

    def _extract_bars(self, data):
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (List) of bars built using the current batch.
        """
        cum_ticks, cum_volume, cum_theta, high_price, low_price, exp_num_ticks, imbalance_array = self._update_counters()

        # Iterate over rows
        list_bars = []
        for row in data.values:
            # Set variables
            cum_ticks += 1
            date_time = row[0]
            price = np.float(row[1])
            volume = row[2]
            cum_volume += volume

            # Update high low prices
            high_price, low_price = self._update_high_low(
                high_price, low_price, price)

            # Imbalance calculations
            signed_tick = self._apply_tick_rule(price)
            imbalance = self._get_imbalance(price, signed_tick, volume)
            imbalance_array.append(imbalance)
            cum_theta += imbalance
            expected_imbalance = self._get_expected_imbalance(
                exp_num_ticks, imbalance_array)

            # Update cache
            self._update_cache(date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_theta, exp_num_ticks,
                               imbalance_array)

            # Check expression for possible bar generation
            if np.abs(cum_theta) > exp_num_ticks * np.abs(expected_imbalance):
                self._create_bars(date_time, price,
                                  high_price, low_price, list_bars)

                self.num_ticks_bar.append(cum_ticks)
                # Expected number of ticks based on formed bars
                exp_num_ticks = ewma(np.array(self.num_ticks_bar[-self.num_prev_bars:], dtype=float),
                                     self.num_prev_bars)[-1]

                # Reset counters
                cum_ticks, cum_volume, cum_theta = 0, 0, 0
                high_price, low_price = -np.inf, np.inf
                self.cache = []

                # Update cache after bar generation (exp_num_ticks was changed after bar generation)
                self._update_cache(date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_theta, exp_num_ticks,
                                   imbalance_array)

        return list_bars

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: Updated cum_ticks, cum_volume, cum_theta, high_price, low_price, exp_num_ticks, imbalance_array.
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
            # expected number of ticks extracted from prev bars
            exp_num_ticks = np.float(latest_entry.exp_num_ticks)
            # array of latest imbalances
            imbalance_array = latest_entry.imbalance_array
        else:
            # Reset counters
            cum_ticks, cum_theta, cum_volume = 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks, imbalance_array = self.exp_num_ticks_init, []

        return cum_ticks, cum_volume, cum_theta, high_price, low_price, exp_num_ticks, imbalance_array

    def _update_cache(self, date_time, price, low_price, high_price, cum_ticks, cum_volume, cum_theta, exp_num_ticks,
                      imbalance_array):
        """
        Update the cache which is used to create a continuous flow of bars from one batch to the next.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param low_price: Lowest price in the period
        :param high_price: Highest price in the period
        :param cum_ticks: Cumulative number of ticks
        :param cum_volume: Cumulative volume (# of contracts)
        :param cum_theta: Cumulative Theta sub t (pg 29)
        :param exp_num_ticks: E[T]
        :param imbalance_array: (numpy array) of the tick imbalances
        """
        cache_data = self.cache_tuple(date_time=date_time, price=price, high=high_price, low=low_price,
                                      cum_ticks=cum_ticks, cum_volume=cum_volume, cum_theta=cum_theta, exp_num_ticks=exp_num_ticks,
                                      imbalance_array=imbalance_array)
        self.cache.append(cache_data)

    def _get_expected_imbalance(self, exp_num_ticks, imbalance_array):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29

        :param exp_num_ticks: The expected number of ticks in the bar
        :param imbalance_array: (numpy array) of the tick imbalances
        :return: expected_imbalance: 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(imbalance_array) < exp_num_ticks:
            # Waiting for array to fill for ewma
            expected_imbalance = np.nan
        else:
            # Expected imbalance per tick
            if self.imbalance_ewma_window is None:
                ewma_window = int(exp_num_ticks)
            else:
                ewma_window = self.imbalance_ewma_window
            expected_imbalance = ewma(
                np.array(imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]

        return expected_imbalance


def get_dollar_imbalance_bars(file_path, num_prev_bars, imbalance_ewma_window=None, exp_num_ticks_init=100000,
                              batch_size=2e7, verbose=True):
    """
    Creates the dollar imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param imbalance_ewma_window: Window size for imbalance calculation
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :return: Dataframe of dollar bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='dollar_imbalance', num_prev_bars=num_prev_bars,
                         imbalance_ewma_window=imbalance_ewma_window, exp_num_ticks_init=exp_num_ticks_init,
                         batch_size=batch_size)
    dollar_imbalance_bars = bars.batch_run(verbose=verbose)

    return dollar_imbalance_bars


def get_volume_imbalance_bars(file_path, num_prev_bars, imbalance_ewma_window=None, exp_num_ticks_init=100000,
                              batch_size=2e7, verbose=True):
    """
    Creates the volume imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param imbalance_ewma_window: Window size for imbalance calculation
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :return: Dataframe of volume bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='volume_imbalance', num_prev_bars=num_prev_bars,
                         imbalance_ewma_window=imbalance_ewma_window, exp_num_ticks_init=exp_num_ticks_init,
                         batch_size=batch_size)
    volume_imbalance_bars = bars.batch_run(verbose=verbose)

    return volume_imbalance_bars


def get_tick_imbalance_bars(file_path, num_prev_bars, imbalance_ewma_window=None, exp_num_ticks_init=100000,
                            batch_size=2e7, verbose=True):
    """
    Creates the tick imbalance bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param imbalance_ewma_window: Window size for imbalance calculation
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :return: Dataframe of tick bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='tick_imbalance', num_prev_bars=num_prev_bars,
                         imbalance_ewma_window=imbalance_ewma_window, exp_num_ticks_init=exp_num_ticks_init,
                         batch_size=batch_size)
    tick_imbalance_bars = bars.batch_run(verbose=verbose)

    return tick_imbalance_bars
