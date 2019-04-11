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


A good blog post to read, which helped us a lot in the implementation here is writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

# Imports
from collections import namedtuple
import numpy as np

from mlfinlab.util.fast_ewma import ewma
from mlfinlab.data_structures.base_bars import BaseBars


class RunBars(BaseBars):
    """
    Contains all of the logic to construct the run bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_run_bars which will create an instance of this
    class and then construct the run bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, file_path, metric, num_prev_bars=3, exp_num_ticks_init=100000,
                 batch_size=2e7):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: "dollar_imbalance"
        :param num_prev_bars: (Int) Window size for E[T]
        :param exp_num_ticks_init: (Int) Initial number of expected ticks
        :param batch_size: (Int) Number of rows to read in from the csv, per batch
        """
        BaseBars.__init__(self, file_path, metric, batch_size)

        # Information bar properties
        self.exp_num_ticks_init = exp_num_ticks_init
        # Expected number of ticks extracted from prev bars
        self.exp_num_ticks = self.exp_num_ticks_init
        self.num_prev_bars = num_prev_bars
        self.num_ticks_bar = []  # List of number of ticks from previous bars

        # Named tuple to help with storing the cache
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'cum_volume',
                                       'cum_theta_buy', 'cum_theta_sell'])
        self.imbalance_array = {'buy': [], 'sell': []}
        self.exp_buy_proportion, self.exp_sell_proportion = np.nan, np.nan

    def _extract_bars(self, data):
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (List) of bars built using the current batch.
        """
        cum_ticks, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price = self._update_counters()

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

            if imbalance > 0:
                self.imbalance_array['buy'].append(imbalance)
                # Set zero to keep buy and sell arrays synced
                self.imbalance_array['sell'].append(0)
                cum_theta_buy += imbalance
            elif imbalance < 0:
                self.imbalance_array['sell'].append(abs(imbalance))
                # Set zero to keep buy and sell arrays synced
                self.imbalance_array['buy'].append(0)
                cum_theta_sell += abs(imbalance)

            if not list_bars and np.isnan(self.exp_buy_proportion):
                self.exp_buy_proportion, self.exp_sell_proportion = self._get_expected_imbalance(
                    self.exp_num_ticks, self.imbalance_array)

            # Update cache
            self._update_cache(date_time, price, low_price, high_price, cum_theta_sell, cum_theta_buy,
                               cum_ticks, cum_volume)

            # Check expression for possible bar generation
            max_proportion = max(self.exp_buy_proportion,
                                 self.exp_sell_proportion)
            if max(cum_theta_buy, cum_theta_sell) > self.exp_num_ticks * max_proportion:
                self._create_bars(date_time, price,
                                  high_price, low_price, list_bars)

                self.num_ticks_bar.append(cum_ticks)
                # Expected number of ticks based on formed bars
                self.exp_num_ticks = ewma(np.array(self.num_ticks_bar[-self.num_prev_bars:], dtype=float),
                                          self.num_prev_bars)[-1]
                self.exp_buy_proportion, self.exp_sell_proportion = self._get_expected_imbalance(
                    self.exp_num_ticks * self.num_prev_bars, self.imbalance_array)

                # Reset counters
                cum_ticks, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0
                high_price, low_price = -np.inf, np.inf
                self.cache = []

                # Update cache after bar generation (exp_num_ticks was changed after bar generation)
                self._update_cache(date_time, price, low_price, high_price, cum_theta_sell, cum_theta_buy,
                                   cum_ticks, cum_volume)
        return list_bars

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: Updated cum_ticks, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks
        """
        # Check flag
        if self.flag and self.cache:
            latest_entry = self.cache[-1]

            # Update variables based on cache
            cum_ticks = int(latest_entry.cum_ticks)
            cum_volume = int(latest_entry.cum_volume)
            low_price = np.float(latest_entry.low)
            high_price = np.float(latest_entry.high)
            # Cumulative buy and sell imbalances for a particular run calculation (theta_t in Prado book)
            cum_theta_buy = np.float(latest_entry.cum_theta_buy)
            cum_theta_sell = np.float(latest_entry.cum_theta_sell)

        else:
            # Reset counters
            cum_ticks, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf

        return cum_ticks, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price

    def _update_cache(self, date_time, price, low_price, high_price, cum_theta_sell, cum_theta_buy,
                      cum_ticks, cum_volume):
        """
        Update the cache which is used to create a continuous flow of bars from one batch to the next.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param low_price: Lowest price in the period
        :param high_price: Highest price in the period
        :param cum_theta_sell: Cumulation of negative signed ticks
        :param cum_theta_buy: Cumulation of positive signed ticks
        :param cum_ticks: Cumulative number of ticks
        """
        cache_data = self.cache_tuple(date_time=date_time, price=price, high=high_price, low=low_price,
                                      cum_ticks=cum_ticks, cum_volume=cum_volume, cum_theta_buy=cum_theta_buy,
                                      cum_theta_sell=cum_theta_sell)
        self.cache.append(cache_data)

    def _get_expected_imbalance(self, window, imbalance_array):
        """
        Calculate the expected imbalance as defined on page 31 and 32.
        :param window: EWMA window for calculation
        :param imbalance_array: numpy array of imbalances [buy, sell]
        :return: expected_buy_proportion and expected_sell_proportion
        """
        if len(imbalance_array['buy']) < self.exp_num_ticks_init:
            # Waiting for array to fill for ewma
            return np.nan, np.nan
        elif len(imbalance_array) < window:
            window = min(len(imbalance_array), window)

        ewma_window = int(window)
        buy_sample = np.array(
            imbalance_array['buy'][-ewma_window:], dtype=float)
        sell_sample = np.array(
            imbalance_array['sell'][-ewma_window:], dtype=float)
        buy_and_sell_imb = sum(buy_sample) + sum(sell_sample)
        exp_buy_proportion = ewma(
            buy_sample, window=ewma_window)[-1] / buy_and_sell_imb
        exp_sell_proportion = ewma(
            sell_sample, window=ewma_window)[-1] / buy_and_sell_imb

        return exp_buy_proportion, exp_sell_proportion


def get_dollar_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                        batch_size=2e7, verbose=True, to_csv=False):
    """
    Creates the dollar run bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :return: DataFrame of dollar bars
    """

    bars = RunBars(file_path=file_path, metric='dollar_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size)
    dollar_run_bars = bars.batch_run(verbose=verbose, to_csv=to_csv)

    return dollar_run_bars


def get_volume_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                        batch_size=2e7, verbose=True, to_csv=False):
    """
    Creates the volume run bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :return: DataFrame of volume bars
    """
    bars = RunBars(file_path=file_path, metric='volume_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size)
    volume_run_bars = bars.batch_run(verbose=verbose, to_csv=to_csv)

    return volume_run_bars


def get_tick_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                      batch_size=2e7, verbose=True, to_csv=False):
    """
    Creates the tick run bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :return: DataFrame of tick bars
    """
    bars = RunBars(file_path=file_path, metric='tick_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size)
    tick_run_bars = bars.batch_run(verbose=verbose, to_csv=to_csv)

    return tick_run_bars
