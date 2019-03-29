"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures: Imbalance Bars

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of tick, volume, and dollar imbalance bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 29) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time
interval sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high
frequency paradigm, Lopez de Prado, et al.
"""

# Imports
from collections import namedtuple
import pandas as pd
import numpy as np
from mlfinlab.util.fast_ewma import ewma


class ImbalanceBars:
    """
    Contains all of the logic to construct the imbalance bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_imbalance_bars which will create an instance of this
    class and then construct the imbalance bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """
    def __init__(self, file_path, metric, exp_num_ticks_init=100000,
                 num_prev_bars=3, num_ticks_ewma_window=20, batch_size=2e7):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: dollar_imbalance.
        :param exp_num_ticks_init: Initial number of expected ticks
        :param num_prev_bars: Number of previous bars to use in calculations
        :param num_ticks_ewma_window: Window size for E[T]
        :param batch_size: Number of rows to read in from the csv, per batch.
        """

        # Base properties
        self.file_path = file_path
        self.metric = metric
        self.exp_num_ticks_init = exp_num_ticks_init
        self.num_prev_bars = num_prev_bars
        self.num_ticks_ewma_window = num_ticks_ewma_window
        self.batch_size = batch_size

        # Batch_run properties
        self.flag = False  # The first flag is false since the first batch doesn't use the cache
        self.cache = []
        self.num_ticks_bar = []  # List of number of ticks from previous bars

        # Extract bars properties
        self.cache_tuple = namedtuple('CacheData', ['date_time', 'price', 'high', 'low',
                                                    'tick_rule', 'cum_ticks', 'cum_theta', 'exp_num_ticks',
                                                    'imbalance_array'])

    def _extract_bars(self, data):
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (List) of bars built using the current batch.
        """
        cum_ticks, cum_theta, high_price, low_price, exp_num_ticks, imbalance_array = self._update_counters()

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
            if price > high_price:
                high_price = price
            if price <= low_price:
                low_price = price

            # Imbalance calculations
            signed_tick, prev_tick_rule = self._apply_tick_rule(price, prev_tick_rule)
            imbalance = self._get_imbalance(price, signed_tick, volume)
            imbalance_array.append(imbalance)
            cum_theta += imbalance
            expected_imbalance = self._get_expected_imbalance(exp_num_ticks, imbalance_array)

            # Update cache
            self._update_cache(date_time, price, low_price, high_price, signed_tick, cum_ticks, cum_theta, exp_num_ticks,
                               imbalance_array)

            # Check expression for possible bar generation
            if np.abs(cum_theta) > exp_num_ticks * np.abs(expected_imbalance):  # pylint: disable=eval-used
                self._create_bars(date_time, price, high_price, low_price, list_bars, cum_ticks)

                # Expected number of ticks based on formed bars
                exp_num_ticks = ewma(np.array(self.num_ticks_bar[-self.num_ticks_ewma_window:], dtype=float),
                                     self.num_ticks_ewma_window)[-1]

                # Reset counters
                cum_ticks, cum_theta = 0, 0
                high_price, low_price = -np.inf, np.inf
                self.cache = []

            # Update cache after bar generation (exp_num_ticks was changed after bar generation)
            self._update_cache(date_time, price, low_price, high_price, signed_tick, cum_ticks, cum_theta, exp_num_ticks,
                               imbalance_array)

        return list_bars

    def _update_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.
        :return: Updated cum_ticks, cum_dollar_value, cum_volume, high_price, low_price, exp_num_ticks, imbalance_array.
        """
        # Check flag
        if self.flag and self.cache:
            latest_entry = self.cache[-1]

            # Update variables based on cache
            cum_ticks = int(latest_entry.cum_ticks)
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
            cum_ticks, cum_theta = 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks, imbalance_array = self.exp_num_ticks_init, []

        return cum_ticks, cum_theta, high_price, low_price, exp_num_ticks, imbalance_array

    def _update_cache(self, date_time, price, low_price, high_price, signed_tick, cum_ticks, cum_theta, exp_num_ticks,
                      imbalance_array):
        """
        Update the cache which is used to create a continuous flow of bars from one batch to the next.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param low_price: Lowest price in the period
        :param high_price: Highest price in the period
        :param signed_tick: The signed tick as defined by the tick rule
        :param cum_ticks: Cumulative number of ticks
        :param cum_theta: Cumulative Theta sub t (pg 29)
        :param exp_num_ticks: E[T]
        :param imbalance_array: (numpy array) of the tick imbalances
        """
        cache_data = self.cache_tuple(date_time, price, high_price, low_price, signed_tick, cum_ticks, cum_theta,
                                      exp_num_ticks, imbalance_array)
        self.cache.append(cache_data)

    def _apply_tick_rule(self, price, prev_tick_rule):
        """
        Applies the tick rule as defined on page 29.

        :param price: Price at time t.
        :param prev_tick_rule: The previous tick rule
        :return: The signed tick as well as the updated previous tick rule.
        """
        if self.cache:
            tick_diff = price - self.cache[-1].price
            prev_tick_rule = self.cache[-1].tick_rule
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
        else:
            signed_tick = prev_tick_rule

        return signed_tick, prev_tick_rule

    def _get_imbalance(self, price, signed_tick, volume):
        """
        Get the imbalance at a point in time, denoted as Theta_t in the book, pg 29.

        :param price: Price at t
        :param signed_tick: signed tick, using the tick rule
        :param volume: Volume traded at t
        :return: Imbalance at time t
        """
        if self.metric == 'tick_imbalance':
            imbalance = signed_tick
        elif self.metric == 'dollar_imbalance':
            imbalance = signed_tick * volume * price
        else:  # volume imbalance
            imbalance = signed_tick * volume

        return imbalance

    def _get_expected_imbalance(self, exp_num_ticks, imbalance_array):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29

        :param exp_num_ticks: The expected number of ticks in the bar
        :param imbalance_array: (numpy array) of the tick imbalances
        :return: expected_imbalance: 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(imbalance_array) < exp_num_ticks:
            expected_imbalance = np.nan  # Waiting for array to fill for ewma
        else:
            # Expected imbalance per tick
            ewma_window = int(exp_num_ticks * self.num_prev_bars)
            expected_imbalance = ewma(np.array(imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]

        return expected_imbalance

    def _create_bars(self, date_time, price, high_price, low_price, list_bars, cum_ticks):
        """
        Given the inputs, construct a bar which has the following fields: date_time, open, high, low, close, cum_ticks.
        These bars are appended to the list_bars list, which is later used to construct the final imbalance bars
        DataFrame.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param high_price: Highest price in the period
        :param low_price: Lowest price in the period
        :param list_bars: List to which we append the bars
        :param cum_ticks: Cumulative number of ticks
        """
        # Create bars
        open_price = self.cache[0].price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price
        self.num_ticks_bar.append(cum_ticks)

        # Update bars
        list_bars.append([date_time, open_price, high_price, low_price, close_price, cum_ticks])

    @staticmethod
    def _assert_csv(test_batch):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.
        """
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1],
                          float), 'price column in csv not float.'
        assert isinstance(test_batch.iloc[0, 2],
                          np.int64), 'volume column in csv not int.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print('csv file, column 0, not a date time format:',
                  test_batch.iloc[0, 0])

    def batch_run(self):
        """
        Reads a csv file in batches and then constructs the financial data structure in the form of a DataFrame.
        The csv file must have only 3 columns: date_time, price, & volume.

        :return: (DataFrame) Financial data structure
        """
        # Read in the first row & assert format
        first_row = pd.read_csv(self.file_path, nrows=1)
        self._assert_csv(first_row)

        print('Reading data in batches:')

        # Read csv in batches
        count = 0
        final_bars = []
        for batch in pd.read_csv(self.file_path, chunksize=self.batch_size):
            print('Batch number:', count)
            list_bars = self._extract_bars(data=batch)

            # Append to bars list
            final_bars += list_bars
            count += 1

            # Set flag to True: notify function to use cache
            self.flag = True

        # Return a DataFrame
        cols = ['date_time', 'open', 'high', 'low', 'close', 'cum_ticks']
        bars_df = pd.DataFrame(final_bars, columns=cols)
        print('Returning bars \n')
        return bars_df


def get_dollar_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the dollar imbalance bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='dollar_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                         num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window,
                         batch_size=batch_size)
    dollar_imbalance_bars = bars.batch_run()

    return dollar_imbalance_bars


def get_volume_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the volume imbalance bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='volume_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                         num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window,
                         batch_size=batch_size)
    volume_imbalance_bars = bars.batch_run()

    return volume_imbalance_bars


def get_tick_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=2e7):
    """
    Creates the tick imbalance bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='tick_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                         num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window,
                         batch_size=batch_size)
    tick_imbalance_bars = bars.batch_run()

    return tick_imbalance_bars
