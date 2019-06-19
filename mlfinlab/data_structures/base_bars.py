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
        :param batch_size: (Int) Number of rows to read in from the csv, per batch.
        """
        # Base properties
        self.file_path = file_path
        self.metric = metric
        self.batch_size = batch_size
        self.prev_tick_rule = 0

        # Batch_run properties
        self.flag = False  # The first flag is false since the first batch doesn't use the cache
        self.cache = []

    def batch_run(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a csv file in batches and then constructs the financial data structure in the form of a DataFrame.
        The csv file must have only 3 columns: date_time, price, & volume.
        :param verbose: (Boolean) Flag whether to print message on each processed batch or not
        :param to_csv: (Boolean) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (Boolean) Path to results file, if to_csv = True

        :return: (DataFrame or None) Financial data structure
        """

        # Read in the first row & assert format
        first_row = pd.read_csv(self.file_path, nrows=1)
        self._assert_csv(first_row)

        if to_csv is True:
            header = True  # if to_csv is True, header should written on the first batch only
            open(output_path, 'w').close()  # clean output csv file

        if verbose:  # pragma: no cover
            print('Reading data in batches:')

        # Read csv in batches
        count = 0
        final_bars = []
        cols = ['date_time', 'open', 'high', 'low', 'close', 'volume']
        for batch in pd.read_csv(self.file_path, chunksize=self.batch_size):
            if verbose:  # pragma: no cover
                print('Batch number:', count)

            list_bars = self._extract_bars(data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else:
                # Append to bars list
                final_bars += list_bars
            count += 1

            # Set flag to True: notify function to use cache
            self.flag = True

        if verbose:  # pragma: no cover
            print('Returning bars \n')

        # Return a DataFrame
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df

        # Processed DataFrame is stored in .csv file, return None
        return None

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

        :param test_batch: (DataFrame) the first row of the dataset.
        """
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

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
        These bars are appended to list_bars, which is later used to construct the final bars DataFrame.

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
        volume = self.cache[-1].cum_volume

        # Update bars
        list_bars.append([date_time, open_price, high_price, low_price, close_price, volume])

    def _apply_tick_rule(self, price):
        """
        Applies the tick rule as defined on page 29.

        :param price: Price at time t
        :return: The signed tick
        """
        if self.cache:
            tick_diff = price - self.cache[-1].price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        return signed_tick

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
        else:  # volume imbalance or volume run
            imbalance = signed_tick * volume

        return imbalance
