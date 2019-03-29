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
from mlfinlab.util.fast_ewma import ewma


class InformationBars:
    def __init__(self, file_path, metric, exp_num_ticks_init=100000, num_prev_bars=3, num_ticks_ewma_window=20,
                 batch_size=2e7):

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


class RunBars(InformationBars):
    def __init__(self, file_path, metric, exp_num_ticks_init=100000,
                 num_prev_bars=3, num_ticks_ewma_window=20, batch_size=2e7):

        InformationBars.__init__(self, file_path, metric, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window,
                                 batch_size)

        # Extract bars properties
        self.cache_tuple = namedtuple('CacheData',
                                      ['date_time', 'price', 'high', 'low', 'tick_rule', 'cum_volume',
                                       'cum_dollar_value',
                                       'cum_ticks', 'cum_theta_buy', 'cum_theta_sell', 'exp_num_ticks',
                                       'imbalance_array'])

    def _update_counters(self):
        # Check flag
        if self.flag and self.cache:
            # Update variables based on cache
            cum_ticks = int(self.cache[-1].cum_ticks)
            cum_dollar_value = np.float(self.cache[-1].cum_dollar_value)
            cum_volume = self.cache[-1].cum_volume
            low_price = np.float(self.cache[-1].low)
            high_price = np.float(self.cache[-1].high)
            # cumulative buy and sell imbalances for a particular run calculation (theta_t in Prado book)
            cum_theta_buy = np.float(self.cache[-1].cum_theta_buy)
            cum_theta_sell = np.float(self.cache[-1].cum_theta_sell)
            # expected number of ticks extracted from prev bars
            exp_num_ticks = np.float(self.cache[-1].exp_num_ticks)
            # array of latest imbalances
            imbalance_array = self.cache[-1].imbalance_array
        else:
            # Reset counters
            cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks = self.exp_num_ticks_init
            # In run bars we need to track both buy and sell imbalance
            imbalance_array = {'buy': [], 'sell': []}

        return cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array

    def _extract_bars(self, data):

        cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell, high_price, low_price, exp_num_ticks, imbalance_array = self._update_counters()

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

            # Calculations
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

            if self.metric == 'tick_run':
                imbalance = tick_rule
            elif self.metric == 'dollar_run':
                imbalance = tick_rule * volume * price
            else:  # volume run
                imbalance = tick_rule * volume

            if imbalance > 0:
                imbalance_array['buy'].append(imbalance)
                # set zero to keep buy and sell arrays synced
                imbalance_array['sell'].append(0)
                cum_theta_buy += imbalance
            elif imbalance < 0:
                imbalance_array['sell'].append(abs(imbalance))
                imbalance_array['buy'].append(0)
                cum_theta_sell += abs(imbalance)

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

            # Check min max
            if price > high_price:
                high_price = price
            if price <= low_price:
                low_price = price

            # Update cache
            cache_data = self.cache_tuple(date_time, price, high_price, low_price, tick_rule, cum_volume,
                                          cum_dollar_value,
                                          cum_ticks, cum_theta_buy, cum_theta_sell, exp_num_ticks, imbalance_array)
            self.cache.append(cache_data)

            # Check expression for possible bar generation
            if max(cum_theta_buy, cum_theta_sell) > exp_num_ticks * max(exp_buy_proportion,
                                                                        exp_sell_proportion):  # pylint: disable=eval-used
                # Create bars
                open_price = self.cache[0].price
                high_price = max(high_price, open_price)
                low_price = min(low_price, open_price)
                close_price = price
                self.num_ticks_bar.append(cum_ticks)
                expected_num_ticks_bar = ewma(
                    np.array(self.num_ticks_bar[-self.num_ticks_ewma_window:], dtype=float),
                    self.num_ticks_ewma_window)[
                    -1]  # expected number of ticks based on formed bars
                # Update bars & Reset counters
                list_bars.append([date_time, open_price, high_price, low_price, close_price,
                                  cum_volume, cum_dollar_value, cum_ticks])
                cum_ticks, cum_dollar_value, cum_volume, cum_theta_buy, cum_theta_sell = 0, 0, 0, 0, 0
                high_price, low_price = -np.inf, np.inf
                exp_num_ticks = expected_num_ticks_bar
                self.cache = []

            # Update cache after bar generation (exp_num_ticks was changed after bar generation)
            cache_data = self.cache_tuple(date_time, price, high_price, low_price, tick_rule, cum_volume,
                                          cum_dollar_value,
                                          cum_ticks, cum_theta_buy, cum_theta_sell, exp_num_ticks, imbalance_array)
            self.cache.append(cache_data)
        return list_bars

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
        cols = ['date_time', 'open', 'high', 'low',
                'close', 'cum_vol', 'cum_dollar', 'cum_ticks']

        bars_df = pd.DataFrame(final_bars, columns=cols)
        print('Returning bars \n')
        return bars_df


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
