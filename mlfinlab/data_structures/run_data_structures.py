"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of tick, volume, and dollar run bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 31) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al
"""

# Imports
from collections import namedtuple
import numpy as np
from mlfinlab.util.fast_ewma import ewma
from mlfinlab.data_structures.information_bars import InformationBars


class RunBars(InformationBars):
    """
    Contains all of the logic to construct the run bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_run_bars which will create an instance of this
    class and then construct the run bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, file_path, metric, exp_num_ticks_init=100000, num_prev_bars=3, num_ticks_ewma_window=20,
                 batch_size=2e7):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: dollar_imbalance.
        :param exp_num_ticks_init: Initial number of expected ticks
        :param num_prev_bars: Number of previous bars to use in calculations
        :param num_ticks_ewma_window: Window size for E[T]
        :param batch_size: Number of rows to read in from the csv, per batch.
        """
        InformationBars.__init__(self, file_path, metric, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window,
                                 batch_size)

        # Extract bars properties
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'tick_rule',
                                       'cum_ticks', 'cum_theta_buy', 'cum_theta_sell', 'exp_num_ticks',
                                       'imbalance_array'])

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: Updated cum_ticks, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array
        """
        # Check flag
        if self.flag and self.cache:
            latest_entry = self.cache[-1]

            # Update variables based on cache
            cum_ticks = int(latest_entry.cum_ticks)
            low_price = np.float(latest_entry.low)
            high_price = np.float(latest_entry.high)
            # Cumulative buy and sell imbalances for a particular run calculation (theta_t in Prado book)
            cum_theta_buy = np.float(latest_entry.cum_theta_buy)
            cum_theta_sell = np.float(latest_entry.cum_theta_sell)
            # Expected number of ticks extracted from prev bars
            exp_num_ticks = np.float(latest_entry.exp_num_ticks)
            # Array of latest imbalances
            imbalance_array = latest_entry.imbalance_array
        else:
            # Reset counters
            cum_ticks, cum_theta_buy, cum_theta_sell = 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks = self.exp_num_ticks_init
            # In run bars we need to track both buy and sell imbalance
            imbalance_array = {'buy': [], 'sell': []}

        return cum_ticks, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array

    def _extract_bars(self, data):
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (List) of bars built using the current batch.
        """
        cum_ticks, cum_theta_buy, cum_theta_sell, high_price, low_price, \
            exp_num_ticks, imbalance_array = self._update_counters()

        # Set the first tick rule with 0
        prev_tick_rule = 0

        # Iterate over rows
        list_bars = []
        for row in data.values:
            # Set variables
            cum_ticks += 1
            date_time = row[0]
            price = np.float(row[1])
            volume = row[2]

            # Update high low prices
            high_price, low_price = self._update_high_low(high_price, low_price, price)

            # Imbalance calculations
            signed_tick, prev_tick_rule = self._apply_tick_rule(price, prev_tick_rule)
            imbalance = self._get_imbalance(price, signed_tick, volume)

            if imbalance > 0:
                imbalance_array['buy'].append(imbalance)
                # set zero to keep buy and sell arrays synced
                imbalance_array['sell'].append(0)
                cum_theta_buy += imbalance
            elif imbalance < 0:
                imbalance_array['sell'].append(abs(imbalance))
                imbalance_array['buy'].append(0)
                cum_theta_sell += abs(imbalance)

            exp_buy_proportion, exp_sell_proportion = self._get_expected_imbalance(exp_num_ticks, imbalance_array)

            # Update cache
            self._update_cache(date_time, price, low_price, high_price, signed_tick, cum_theta_sell, cum_theta_buy,
                               cum_ticks, exp_num_ticks, imbalance_array)

            # Check expression for possible bar generation
            max_proportion = max(exp_buy_proportion, exp_sell_proportion)
            if max(cum_theta_buy, cum_theta_sell) > exp_num_ticks * max_proportion:
                self._create_bars(date_time, price, high_price, low_price, list_bars, cum_ticks)

                # Expected number of ticks based on formed bars
                exp_num_ticks = ewma(np.array(self.num_ticks_bar[-self.num_ticks_ewma_window:], dtype=float),
                                     self.num_ticks_ewma_window)[-1]

                # Reset counters
                cum_ticks, cum_theta_buy, cum_theta_sell = 0, 0, 0
                high_price, low_price = -np.inf, np.inf
                self.cache = []

                # Update cache after bar generation (exp_num_ticks was changed after bar generation)
                self._update_cache(date_time, price, low_price, high_price, signed_tick, cum_theta_sell, cum_theta_buy,
                                   cum_ticks, exp_num_ticks, imbalance_array)
        return list_bars

    def _update_cache(self, date_time, price, low_price, high_price, signed_tick, cum_theta_sell, cum_theta_buy,
                      cum_ticks, exp_num_ticks, imbalance_array):
        """
        Update the cache which is used to create a continuous flow of bars from one batch to the next.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param low_price: Lowest price in the period
        :param high_price: Highest price in the period
        :param signed_tick: The signed tick as defined by the tick rule
        :param cum_theta_sell: Cumulation of negative signed ticks
        :param cum_theta_buy: Cumulation of positive signed ticks
        :param cum_ticks: Cumulative number of ticks
        :param exp_num_ticks: E{T}
        :param imbalance_array: (numpy array) of the tick imbalances
        """
        cache_data = self.cache_tuple(date_time, price, high_price, low_price, signed_tick,
                                      cum_ticks, cum_theta_buy, cum_theta_sell, exp_num_ticks, imbalance_array)
        self.cache.append(cache_data)

    def _get_expected_imbalance(self, exp_num_ticks, imbalance_array):
        """
        Calculate the expected imbalance as defined on page 31 and 32.

        :param exp_num_ticks: Expected number of ticks
        :param imbalance_array: numpy array of imbalances [buy, sell]
        :return: expected_buy_proportion and expected_sell_proportion
        """
        if len(imbalance_array['buy']) < exp_num_ticks:
            # waiting for array to fill for ewma
            exp_buy_proportion, exp_sell_proportion = np.nan, np.nan
        else:
            # expected imbalance per tick
            ewma_window = int(exp_num_ticks * self.num_prev_bars)
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


def get_dollar_run_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the dollar run bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """

    bars = RunBars(file_path=file_path, metric='dollar_run', exp_num_ticks_init=exp_num_ticks_init,
                   num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window,
                   batch_size=batch_size)
    dollar_run_bars = bars.batch_run()

    return dollar_run_bars


def get_volume_run_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the volume run bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    bars = RunBars(file_path=file_path, metric='volume_run', exp_num_ticks_init=exp_num_ticks_init,
                   num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window,
                   batch_size=batch_size)
    volume_run_bars = bars.batch_run()

    return volume_run_bars


def get_tick_run_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the tick run bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window to estimate expected number of ticks in a bar based on previous bars
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    bars = RunBars(file_path=file_path, metric='tick_run', exp_num_ticks_init=exp_num_ticks_init,
                   num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window,
                   batch_size=batch_size)
    tick_run_bars = bars.batch_run()

    return tick_run_bars
