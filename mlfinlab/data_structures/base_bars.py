"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseBars(ABC):
    """
    Abstract base class which contains the structure which is shared between the various standard and information
    driven bars. There are some methods contained in here that would only be applicable to information bars but
    they are included here so as to avoid a complicated nested class structure.
    """
    def __init__(self, file_path, metric, batch_size=2e7):
        """
        Constructor

        :param file_path: (String) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: Number of rows to read in from the csv, per batch.
        """
        # Base properties
        self.file_path = file_path
        self.metric = metric
        self.batch_size = batch_size

        # Batch_run properties
        self.flag = False  # The first flag is false since the first batch doesn't use the cache
        self.cache = []

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
        cols = ['date_time', 'open', 'high', 'low', 'close']
        bars_df = pd.DataFrame(final_bars, columns=cols)
        print('Returning bars \n')
        return bars_df

    @abstractmethod
    def _extract_bars(self, data):
        """
        This method is required by all the bar types and is used to create the desired bars.
        :param data: (DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (List) of bars built using the current batch.
        """

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

    @staticmethod
    def _update_high_low(high_price, low_price, price):
        """
        Update the high and low prices using the current price.

        :param high_price: Current high price that needs to be updated
        :param low_price: Current low price that needs to be updated
        :param price: Current price
        :return: Updated high and low prices
        """
        if price > high_price:
            high_price = price

        if price <= low_price:
            low_price = price

        return high_price, low_price

    def _create_bars(self, date_time, price, high_price, low_price, list_bars):
        """
        Given the inputs, construct a bar which has the following fields: date_time, open, high, low, close.
        These bars are appended to the list_bars list, which is later used to construct the final bars
        DataFrame.

        :param date_time: Timestamp of the bar
        :param price: The current price
        :param high_price: Highest price in the period
        :param low_price: Lowest price in the period
        :param list_bars: List to which we append the bars
        """
        # Create bars
        open_price = self.cache[0].price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price

        # Update bars
        list_bars.append([date_time, open_price, high_price, low_price, close_price])

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
        if self.metric == 'tick_imbalance' or self.metric == 'tick_run':
            imbalance = signed_tick
        elif self.metric == 'dollar_imbalance' or self.metric == 'dollar_run':
            imbalance = signed_tick * volume * price
        else:  # volume imbalance | run
            imbalance = signed_tick * volume

        return imbalance
