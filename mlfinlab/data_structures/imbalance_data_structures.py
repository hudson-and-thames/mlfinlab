"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of tick, volume, and dollar imbalance bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018, pg 25)
to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval sampling.
A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm, Lopez de Prado, et al
"""

# Imports
from collections import namedtuple
import pandas as pd
import numpy as np
from mlfinlab.data_structures.fast_ewma import ewma


# Todo: Check docstring params are correct

class ImbalanceBars:
    def __init__(self, file_path, metric, exp_num_ticks_init=100000,
                 num_prev_bars=3, num_ticks_ewma_window=20, batch_size=2e7):

        # base properties
        self.file_path = file_path
        self.metric = metric
        self.exp_num_tick_init = exp_num_ticks_init
        self.num_prev_bars = num_prev_bars
        self.num_ticks_ewma_window = num_ticks_ewma_window
        self.batch_size = batch_size

        # batch_run properties
        self.flag = False  # The first flag is false since the first batch doesn't use the cache
        self.cache = []
        self.num_ticks_bar = None

        # extract bars properties
        self.cache_tuple = namedtuple('CacheData', ['date_time', 'price', 'high', 'low',
                                                    'tick_rule', 'cum_volume', 'cum_dollar_value',
                                                    'cum_ticks', 'cum_theta', 'exp_num_ticks',
                                                    'imbalance_array'])

    def _get_updated_counters(self):
        """
        Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

        :return: (Dictionary) updated counters - keys: cum_ticks, cum_dollar_value, cum_volume,
                high_price, low_price, exp_num_ticks, imbalance_array.
        """
        # Check flag
        if self.flag and self.cache:
            # Update variables based on cache
            cum_ticks = int(self.cache[-1].cum_ticks)
            cum_dollar_value = np.float(self.cache[-1].cum_dollar_value)
            cum_volume = self.cache[-1].cum_volume
            low_price = np.float(self.cache[-1].low)
            high_price = np.float(self.cache[-1].high)
            # cumulative imbalance for a particular imbalance calculation (theta_t in Prado book)
            cum_theta = np.float(self.cache[-1].cum_theta)
            # expected number of ticks extracted from prev bars
            exp_num_ticks = np.float(self.cache[-1].exp_num_ticks)
            # array of latest imbalances
            imbalance_array = self.cache[-1].imbalance_array
        else:
            # Reset counters
            cum_ticks, cum_dollar_value, cum_volume, cum_theta = 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks, imbalance_array = self.exp_num_tick_init, []

        # Create a dictionary to hold the counters.
        counters = {'cum_ticks': cum_ticks, 'cum_dollar_value': cum_dollar_value, 'cum_volume': cum_volume,
                    'cum_theta': cum_theta, 'high_price': high_price, 'low_price': low_price,
                    'exp_num_ticks': exp_num_ticks, 'imbalance_array': imbalance_array}

        return counters

    def _extract_bars(self, data):
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: Contains 3 columns - date_time, price, and volume.
        :return: The financial data structure with the cache of short term history.
        """

        # Named tuple for cache
        if not self.cache:
            prev_tick_rule = 0  # set the first tick rule with 0
            self.num_ticks_bar = []  # array of number of ticks from previous bars

        list_bars = []

        # Todo: should counter be an object of its own?
        counters = self._get_updated_counters()
        cum_ticks, cum_dollar_value, cum_volume, cum_theta, \
        high_price, low_price, exp_num_ticks, imbalance_array = counters.values()

        # Iterate over rows
        for row in data.values:
            # Set variables
            date_time = row[0]
            price = np.float(row[1])
            volume = row[2]

            # Calculations
            cum_ticks += 1
            dollar_value = price * volume
            cum_dollar_value = cum_dollar_value + dollar_value
            cum_volume = cum_volume + volume

            # Imbalance calculations
            try:
                tick_diff = price - self.cache[-1].price
                prev_tick_rule = self.cache[-1].tick_rule
            except IndexError:
                tick_diff = 0

            tick_rule = np.sign(tick_diff) if tick_diff != 0 else prev_tick_rule

            if self.metric == 'tick_imbalance':
                imbalance = tick_rule
            elif self.metric == 'dollar_imbalance':
                imbalance = tick_rule * volume * price
            else:
                # volume imbalance (ok to have else here since metric is not user defined)
                imbalance = tick_rule * volume

            imbalance_array.append(imbalance)
            cum_theta += imbalance

            if len(imbalance_array) < exp_num_ticks:
                exp_tick_imb = np.nan  # Waiting for array to fill for ewma
            else:
                # Expected imbalance per tick
                ewma_window = int(exp_num_ticks * self.num_prev_bars)
                exp_tick_imb = ewma(np.array(imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]

            # Check min max
            if price > high_price:
                high_price = price
            if price <= low_price:
                low_price = price

            # Update cache
            cache_data = self.cache_tuple(date_time, price, high_price, low_price, tick_rule, cum_volume,
                                          cum_dollar_value, cum_ticks, cum_theta, exp_num_ticks, imbalance_array)
            self.cache.append(cache_data)

            # Check expression for possible bar generation
            if np.abs(cum_theta) > exp_num_ticks * np.abs(exp_tick_imb):  # pylint: disable=eval-used
                # Create bars
                open_price = self.cache[0].price
                high_price = max(high_price, open_price)
                low_price = min(low_price, open_price)
                close_price = price
                self.num_ticks_bar.append(cum_ticks)

                # Expected number of ticks based on formed bars
                expected_num_ticks_bar = ewma(np.array(self.num_ticks_bar[-self.num_ticks_ewma_window:], dtype=float),
                                              self.num_ticks_ewma_window)[-1]

                # Update bars & Reset counters
                list_bars.append([date_time, open_price, high_price, low_price, close_price,
                                  cum_volume, cum_dollar_value, cum_ticks])
                cum_ticks, cum_dollar_value, cum_volume, cum_theta = 0, 0, 0, 0
                high_price, low_price = -np.inf, np.inf
                exp_num_ticks = expected_num_ticks_bar
                self.cache = []  # Reset cache

            # Update cache after bar generation (exp_num_ticks was changed after bar generation)
            cache_data = self.cache_tuple(date_time, price, high_price, low_price, tick_rule, cum_volume,
                                          cum_dollar_value, cum_ticks, cum_theta, exp_num_ticks, imbalance_array)
            self.cache.append(cache_data)

        return list_bars

    @staticmethod
    def _assert_csv(test_batch):
        """
        Tests that the csv file read has the format: date_time, price, & volume.
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
        cols = ['date_time', 'open', 'high', 'low', 'close', 'cum_vol', 'cum_dollar', 'cum_ticks']
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
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
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
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
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
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    bars = ImbalanceBars(file_path=file_path, metric='tick_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                         num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window,
                         batch_size=batch_size)
    tick_imbalance_bars = bars.batch_run()

    return tick_imbalance_bars
