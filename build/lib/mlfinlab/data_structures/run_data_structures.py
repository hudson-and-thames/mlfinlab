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

    def __init__(self, file_path, metric, num_prev_bars=3, exp_num_ticks_init=100000, batch_size=2e7):
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
        self.num_ticks_bar = {'cum_ticks': [],
                              'buy_proportion': []}  # Dict of number of ticks, number of buy ticks from previous bars

        # Named tuple to help with storing the cache
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'cum_ticks', 'buy_ticks', 'cum_volume',
                                       'cum_theta_buy', 'cum_theta_sell'])
        self.imbalance_array = {'buy': [], 'sell': []}
        self.warm_up = True  # boolean flag for warm-up period
        self.exp_imbalance = {'buy': np.nan, 'sell': np.nan}
        self.exp_buy_ticks_proportion = np.nan

    def _extract_bars(self, data):
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (List) of bars built using the current batch.
        """
        cum_ticks, buy_ticks, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price = self._update_counters()

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
                cum_theta_buy += imbalance
                buy_ticks += 1
            elif imbalance < 0:
                self.imbalance_array['sell'].append(abs(imbalance))
                cum_theta_sell += abs(imbalance)

            imbalances_are_counted_flag = np.isnan([self.exp_imbalance['buy'], self.exp_imbalance[
                'sell']]).any()  # flag indicating that both buy and sell imbalances are counted
            if not list_bars and imbalances_are_counted_flag:
                self.exp_imbalance['buy'] = self._get_expected_imbalance(self.exp_num_ticks,
                                                                         self.imbalance_array['buy'])
                self.exp_imbalance['sell'] = self._get_expected_imbalance(self.exp_num_ticks,
                                                                          self.imbalance_array['sell'])
                if bool(np.isnan([self.exp_imbalance['buy'], self.exp_imbalance['sell']]).any()) is False:
                    self.exp_buy_ticks_proportion = buy_ticks / cum_ticks
                    cum_theta_buy, cum_theta_sell = 0, 0  # reset theta after warm-up period
                    self.warm_up = False

            # Update cache
            self._update_cache(date_time, price, low_price, high_price, cum_theta_sell, cum_theta_buy,
                               cum_ticks, buy_ticks, cum_volume)

            # Check expression for possible bar generation
            max_proportion = max(self.exp_imbalance['buy'] * self.exp_buy_ticks_proportion,
                                 self.exp_imbalance['sell'] * (1 - self.exp_buy_ticks_proportion))
            if max(cum_theta_buy, cum_theta_sell) > self.exp_num_ticks * max_proportion and self.warm_up is False:
                self._create_bars(date_time, price,
                                  high_price, low_price, list_bars)

                self.num_ticks_bar['cum_ticks'].append(cum_ticks)
                self.num_ticks_bar['buy_proportion'].append(buy_ticks / cum_ticks)
                # Expected number of ticks based on formed bars
                self.exp_num_ticks = ewma(np.array(self.num_ticks_bar['cum_ticks'][-self.num_prev_bars:], dtype=float),
                                          self.num_prev_bars)[-1]
                # Expected buy ticks proportion based on formed bars
                self.exp_buy_ticks_proportion = \
                    ewma(np.array(self.num_ticks_bar['buy_proportion'][-self.num_prev_bars:], dtype=float),
                         self.num_prev_bars)[-1]
                self.exp_imbalance['buy'] = self._get_expected_imbalance(self.exp_num_ticks * self.num_prev_bars,
                                                                         self.imbalance_array['buy'])
                self.exp_imbalance['sell'] = self._get_expected_imbalance(self.exp_num_ticks * self.num_prev_bars,
                                                                          self.imbalance_array['sell'])

                # Reset counters
                cum_ticks, buy_ticks, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
                high_price, low_price = -np.inf, np.inf
                self.cache = []

                # Update cache after bar generation (exp_num_ticks was changed after bar generation)
                self._update_cache(date_time, price, low_price, high_price, cum_theta_sell, cum_theta_buy,
                                   cum_ticks, buy_ticks, cum_volume)
        return list_bars

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: Updated cum_ticks, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price
        """
        # Check flag
        if self.flag and self.cache:
            latest_entry = self.cache[-1]

            # Update variables based on cache
            cum_ticks = int(latest_entry.cum_ticks)
            buy_ticks = int(latest_entry.buy_ticks)
            cum_volume = int(latest_entry.cum_volume)
            low_price = np.float(latest_entry.low)
            high_price = np.float(latest_entry.high)
            # Cumulative buy and sell imbalances for a particular run calculation (theta_t in Prado book)
            cum_theta_buy = np.float(latest_entry.cum_theta_buy)
            cum_theta_sell = np.float(latest_entry.cum_theta_sell)

        else:
            # Reset counters
            cum_ticks, buy_ticks, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf

        return cum_ticks, buy_ticks, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price

    def _update_cache(self, date_time, price, low_price, high_price, cum_theta_sell, cum_theta_buy,
                      cum_ticks, buy_ticks, cum_volume):
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
                                      cum_ticks=cum_ticks, buy_ticks=buy_ticks, cum_volume=cum_volume,
                                      cum_theta_buy=cum_theta_buy,
                                      cum_theta_sell=cum_theta_sell)
        self.cache.append(cache_data)

    def _get_expected_imbalance(self, window, imbalance_array):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
        :param window: EWMA window for calculation
        :param imbalance_array: (numpy array) of the tick imbalances
        :return: expected_imbalance: 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(imbalance_array) < self.exp_num_ticks_init:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(min(len(imbalance_array), window))

        if np.isnan(ewma_window):
            # return nan, wait until len(imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]

        return expected_imbalance


def get_dollar_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                        batch_size=2e7, verbose=True, to_csv=False, output_path=None):
    """
    Creates the dollar run bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of dollar bars
    """

    bars = RunBars(file_path=file_path, metric='dollar_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size)
    dollar_run_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return dollar_run_bars


def get_volume_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                        batch_size=2e7, verbose=True, to_csv=False, output_path=None):
    """
    Creates the volume run bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of volume bars
    """
    bars = RunBars(file_path=file_path, metric='volume_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size)
    volume_run_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return volume_run_bars


def get_tick_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                      batch_size=2e7, verbose=True, to_csv=False, output_path=None):
    """
    Creates the tick run bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of tick bars
    """
    bars = RunBars(file_path=file_path, metric='tick_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size)
    tick_run_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return tick_run_bars
