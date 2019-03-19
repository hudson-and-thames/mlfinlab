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
import pandas as pd
import numpy as np
from mlfinlab.data_structures.fast_ewma import ewma


def _update_counters(cache, flag, exp_num_ticks_init):
    """
    Updates the counters by resetting them or making use of the cache to update them based on a previous batch.

    :param cache: Contains information from the previous batch that is relevant in this batch.
    :param flag: A flag which signals to use the cache.
    :param exp_num_ticks: Expected number of ticks per bar
    :param ewma_window: ewma_window to estimate imbalance
    :return: Updated counters - cum_ticks, cum_dollar_value, cum_volume, high_price, low_price, exp_num_ticks, imbalance_array
    """
    # Check flag
    if flag and cache:
        # Update variables based on cache
        cum_ticks = int(cache[-1][6])
        cum_dollar_value = np.float(cache[-1][5])
        cum_volume = cache[-1][4]
        low_price = np.float(cache[-1][2])
        high_price = np.float(cache[-1][3])
        # cumulative imbalances
        cum_dollar_imb = np.float(cache[-1][7])
        cum_tick_imb = np.float(cache[-1][8])
        cum_volume_imb = np.float(cache[-1][9])
        # expected number of ticks extracted from prev bars
        exp_num_ticks = np.float(cache[-1][10])
        imbalance_array = cache[-1][11]  # array of latest imbalances
    else:
        # Reset counters
        cum_ticks, cum_dollar_value, cum_volume, cum_dollar_imb, cum_tick_imb, cum_volume_imb = 0, 0, 0, 0, 0, 0
        high_price, low_price = -np.inf, np.inf
        exp_num_ticks, imbalance_array = exp_num_ticks_init, []

    return cum_ticks, cum_dollar_value, cum_volume, cum_dollar_imb, cum_tick_imb, cum_volume_imb, high_price, low_price, exp_num_ticks, imbalance_array


def _extract_bars(data, metric, exp_num_ticks_init=100000, num_prev_bars=3, num_ticks_ewma_window=20,
                  cache=None, flag=False, prev_price=None, num_ticks_bar=None, prev_tick_rule=None):
    """
    For loop which compiles the various imbalance bars: dollar, volume, or tick.

    :param data: Contains 3 columns - date_time, price, and volume.
    :param metric: dollar_imbalance, volume_imbalance or tick_imbalance
    :param exp_num_ticks_init: initial guess of number of ticks in imbalance bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :param num_ticks_ewma_window: EWMA window to estimate expected number of ticks in a bar from based on previous bars
    :param cache: contains information from the previous batch that is relevant in this batch.
    :param flag: A flag which signals to use the cache.
    :param prev_price: Previous price (we need previous batch price for price diff)
    :param num_ticks_bar: Expected number of ticks per bar used to estimate the next bar
    :param prev_tick_rule: Previous tick rule (if price_diff == 0 => use previous tick rule)
    :return: The financial data structure with the cache of short term history.
    """
    if cache is None:
        cache = []

    if num_ticks_bar is None:  # array of number of ticks from previous bars
        num_ticks_bar = []
    if prev_tick_rule is None:
        prev_tick_rule = 0

    list_bars = []
    cum_ticks, cum_dollar_value, cum_volume, cum_dollar_imb, cum_tick_imb, cum_volume_imb, \
        high_price, low_price, exp_num_ticks, imbalance_array = _update_counters(
            cache, flag, exp_num_ticks_init)

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
        if prev_price is None:
            tick_diff = 0
        else:
            tick_diff = price - prev_price

        if tick_diff > 0:
            tick_rule = 1.0
        elif tick_diff < 0:
            tick_rule = -1.0
        else:
            tick_rule = prev_tick_rule

        dollar_imb = tick_rule * volume * price
        volume_imb = tick_rule * volume
        # cumulative dollar imbalance up till now (reset on every imbalance bar)

        cum_tick_imb += tick_rule
        cum_dollar_imb += dollar_imb
        cum_volume_imb += volume_imb

        if metric == 'tick_imbalance':
            imbalance_array.append(tick_rule)  # latest relevant imbalances
            theta_imb = cum_tick_imb
        elif metric == 'dollar_imbalance':
            imbalance_array.append(dollar_imb)
            theta_imb = cum_dollar_imb
        elif metric == 'volume_imbalance':
            imbalance_array.append(volume_imb)
            theta_imb = cum_volume_imb

        imb_flag = False
        if len(imbalance_array) < exp_num_ticks:
            exp_tick_imb = np.nan  # waiting for array to fill for ewma
        else:
            # expected imbalance per tick
            ewma_window = int(exp_num_ticks * num_prev_bars)
            exp_tick_imb = ewma(
                np.array(imbalance_array[-ewma_window:], dtype=float), window=ewma_window)[-1]
            if np.abs(theta_imb) > exp_num_ticks * np.abs(exp_tick_imb):
                imb_flag = True

        # Check min max
        if price > high_price:
            high_price = price
        if price <= low_price:
            low_price = price

        prev_price = price  # update prev_price
        prev_tick_rule = tick_rule

        #print(date_time, cum_dollar_imb, exp_num_ticks, exp_tick_imb)
        # If threshold reached then take a sample
        if imb_flag is True:   # pylint: disable=eval-used
            # Create bars
            open_price = cache[0][1]
            low_price = min(low_price, open_price)
            close_price = price
            num_ticks_bar.append(cum_ticks)
            expected_num_ticks_bar = ewma(
                np.array(num_ticks_bar[-num_ticks_ewma_window:], dtype=float), num_ticks_ewma_window)[-1]  # expected number of ticks based on formed bars
            # Update bars & Reset counters
            list_bars.append([date_time, open_price, high_price, low_price, close_price,
                              cum_volume, cum_dollar_value, cum_ticks])
            cum_ticks, cum_dollar_value, cum_volume, cum_dollar_imb, cum_tick_imb, cum_volume_imb = 0, 0, 0, 0, 0, 0
            high_price, low_price = -np.inf, np.inf
            exp_num_ticks = expected_num_ticks_bar

        # Update cache
        cache.append([date_time, price, low_price, high_price,
                      cum_volume, cum_dollar_value, cum_ticks, cum_dollar_imb, cum_tick_imb, cum_volume_imb, exp_num_ticks, imbalance_array])
    return list_bars, cache, num_ticks_bar, prev_price, prev_tick_rule


def _assert_dataframe(test_batch):
    """
    Tests that the csv file read has the format: date_time, price, & volume.
    If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

    :param test_batch: DataFrame which will be tested.
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


def _batch_run(file_path, metric, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=20000000):
    """
    Reads a csv file in batches and then constructs the financial data structure in the form of a DataFrame.

    The csv file must have only 3 columns: date_time, price, & volume.

    :param file_path: File path pointing to csv data.
    :param metric: tick_imbalance, dollar_imbalance or volume_imbalance
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Financial data structure
    """
    print('Reading data in batches:')

    # Variables
    count = 0
    flag = False  # The first flag is false since the first batch doesn't use the cache
    cache = None
    prev_price = None
    prev_tick_rule = None
    num_ticks_bar = None
    final_bars = []

    # Read in the first row & assert format
    _assert_dataframe(pd.read_csv(file_path, nrows=1))

    # Read csv in batches
    for batch in pd.read_csv(file_path, chunksize=batch_size):

        print('Batch number:', count)
        list_bars, cache, num_ticks_bar, prev_price, prev_tick_rule = _extract_bars(
            data=batch, metric=metric, exp_num_ticks_init=exp_num_ticks_init, num_prev_bars=num_prev_bars,
            num_ticks_ewma_window=num_ticks_ewma_window, cache=cache, flag=flag, prev_price=prev_price,
            num_ticks_bar=num_ticks_bar, prev_tick_rule=prev_tick_rule)
        # Append to bars list
        final_bars += list_bars
        count += 1

        # Set flag to True: notify function to use cache
        flag = True

    # Return a DataFrame
    cols = ['date_time', 'open', 'high', 'low',
            'close', 'cum_vol', 'cum_dollar', 'cum_ticks']
    bars_df = pd.DataFrame(final_bars, columns=cols)
    print('Returning bars \n')
    return bars_df


def get_dollar_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=20000000):
    """
    Creates the dollar imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='dollar_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)


def get_volume_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=20000000):
    """
    Creates the volume imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='volume_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)


def get_tick_imbalance_bars(file_path, exp_num_ticks_init, num_prev_bars, num_ticks_ewma_window, batch_size=20000000):
    """
    Creates the tick imbalace bars: date_time, open, high, low, close, cum_vol, cum_dollar, and cum_ticks.
    :param file_path: File path pointing to csv data.
    :param exp_num_ticks_init: initial expetected number of ticks per bar
    :param num_prev_bars: Number of previous bars used for EWMA window (window=num_prev_bars * bar length)
                          for estimating expected imbalance (tick, volume or dollar)
    :num_ticks_ewma_window: EWMA window for expected number of ticks calculations
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :return: Dataframe of dollar bars
    """
    return _batch_run(file_path=file_path, metric='tick_imbalance', exp_num_ticks_init=exp_num_ticks_init,
                      num_prev_bars=num_prev_bars, num_ticks_ewma_window=num_ticks_ewma_window, batch_size=batch_size)
