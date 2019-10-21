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

    def __init__(self, file_path, metric, num_prev_bars=3, exp_num_ticks_init=100000, batch_size=2e7,
                 analyse_thresholds=False):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: "dollar_imbalance"
        :param num_prev_bars: (Int) Window size for E[T]
        :param exp_num_ticks_init: (Int) Initial number of expected ticks
        :param batch_size: (Int) Number of rows to read in from the csv, per batch
        :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
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
        self.cum_theta_buy, self.cum_theta_sell, self.buy_ticks = 0, 0, 0
        self.imbalance_array = {'buy': [], 'sell': []}
        self.warm_up = True  # boolean flag for warm-up period
        self.exp_imbalance = {'buy': np.nan, 'sell': np.nan}
        self.exp_buy_ticks_proportion = np.nan

        self.analyse_thresholds = analyse_thresholds
        self.bars_thresholds = []  # Array of dicts:
        # {'timestamp': value, 'cum_theta_buy': value, 'cum_theta_sell': value,
        # 'exp_num_ticks': value, 'exp_imbalance_buy': value, 'exp_imbalance_sell': value,
        # 'exp_buy_ticks_proportion': value}

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_ticks, self.cum_dollar_value, self.cum_volume, = 0, 0, 0
        self.cum_theta_buy, self.cum_theta_sell, self.buy_ticks = 0, 0, 0

    def _extract_bars(self, data):
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

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

            if imbalance > 0:
                self.imbalance_array['buy'].append(imbalance)
                self.cum_theta_buy += imbalance
                self.buy_ticks += 1
            elif imbalance < 0:
                self.imbalance_array['sell'].append(abs(imbalance))
                self.cum_theta_sell += abs(imbalance)

            imbalances_are_counted_flag = np.isnan([self.exp_imbalance['buy'], self.exp_imbalance[
                'sell']]).any()  # flag indicating that both buy and sell imbalances are counted
            if not list_bars and imbalances_are_counted_flag:
                self.exp_imbalance['buy'] = self._get_expected_imbalance(self.exp_num_ticks,
                                                                         self.imbalance_array['buy'])
                self.exp_imbalance['sell'] = self._get_expected_imbalance(self.exp_num_ticks,
                                                                          self.imbalance_array['sell'])
                if bool(np.isnan([self.exp_imbalance['buy'], self.exp_imbalance['sell']]).any()) is False:
                    self.exp_buy_ticks_proportion = self.buy_ticks / self.cum_ticks
                    self.cum_theta_buy, self.cum_theta_sell, self.buy_ticks = 0, 0, 0  # reset thetas and buy_ticks after warm-up period
                    self.warm_up = False

            # Check expression for possible bar generation
            max_proportion = max(self.exp_imbalance['buy'] * self.exp_buy_ticks_proportion,
                                 self.exp_imbalance['sell'] * (1 - self.exp_buy_ticks_proportion))

            if self.analyse_thresholds is True:
                logs_dict = {'timestamp': date_time, 'cum_theta_buy': self.cum_theta_buy,
                             'cum_theta_sell': self.cum_theta_sell,
                             'exp_num_ticks': self.exp_num_ticks, 'exp_imbalance_buy': self.exp_imbalance['buy'],
                             'exp_imbalance_sell': self.exp_imbalance['sell'],
                             'exp_buy_ticks_proportion': self.exp_buy_ticks_proportion}
                self.bars_thresholds.append(logs_dict)

            self.prev_price = price  # Update previous price used for tick rule calculations

            if max(self.cum_theta_buy,
                   self.cum_theta_sell) > self.exp_num_ticks * max_proportion and self.warm_up is False:
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                self.num_ticks_bar['cum_ticks'].append(self.cum_ticks)
                self.num_ticks_bar['buy_proportion'].append(self.buy_ticks / self.cum_ticks)
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
                self._reset_cache()
        return list_bars

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
                        batch_size=2e7, analyse_thresholds=False, verbose=True, to_csv=False, output_path=None):
    """
    Creates the dollar run bars: date_time, open, high, low, close, volume.

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

    bars = RunBars(file_path=file_path, metric='dollar_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    dollar_run_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return dollar_run_bars, bars.bars_thresholds


def get_volume_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                        batch_size=2e7, analyse_thresholds=False, verbose=True, to_csv=False, output_path=None):
    """
    Creates the volume run bars: date_time, open, high, low, close, volume.

    :param file_path: File path pointing to csv data.
    :param num_prev_bars: Number of previous bars used for EWMA window expected # of ticks
    :param exp_num_ticks_init: initial expected number of ticks per bar
    :param batch_size: The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: Print out batch numbers (True or False)
    :param to_csv: Save bars to csv after every batch run (True or False)
    :param analyse_thresholds: (Boolean) Flag to save  and return thresholds used to sample imbalance bars
    :param output_path: Path to csv file, if to_csv is True
    :return: DataFrame of volume bars
    """
    bars = RunBars(file_path=file_path, metric='volume_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    volume_run_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return volume_run_bars, bars.bars_thresholds


def get_tick_run_bars(file_path, num_prev_bars, exp_num_ticks_init=100000,
                      batch_size=2e7, analyse_thresholds=False, verbose=True, to_csv=False, output_path=None):
    """
    Creates the tick run bars: date_time, open, high, low, close, volume.

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
    bars = RunBars(file_path=file_path, metric='tick_run', num_prev_bars=num_prev_bars,
                   exp_num_ticks_init=exp_num_ticks_init, batch_size=batch_size, analyse_thresholds=analyse_thresholds)
    tick_run_bars = bars.batch_run(
        verbose=verbose, to_csv=to_csv, output_path=output_path)

    return tick_run_bars, bars.bars_thresholds
