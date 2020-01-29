"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

Time bars generation logic
"""

# Imports
import numpy as np

from mlfinlab.data_structures.base_bars import BaseBars


# pylint: disable=too-many-instance-attributes
class TimeBars(BaseBars):
    """
    Contains all of the logic to construct the time bars. This class shouldn't be used directly.
    Use get_time_bars instead
    """

    def __init__(self, file_path_or_df, resolution, num_units, batch_size=20000000):

        BaseBars.__init__(self, file_path_or_df, metric=None, batch_size=batch_size)

        # Threshold at which to sample (in seconds)
        self.time_bar_thresh_mapping = {'D': 86400, 'H': 3600, 'MIN': 60, 'S': 1}  # Number of seconds
        assert resolution in self.time_bar_thresh_mapping, "{} resolution is not implemented".format(resolution)
        self.resolution = resolution  # Type of bar resolution: 'D', 'H', 'MIN', 'S'
        self.num_units = num_units  # Number of days/minutes/...
        self.threshold = self.num_units * self.time_bar_thresh_mapping[self.resolution]
        self.timestamp = None  # Current bar timestamp

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for time bars
        """
        self.open_price = None
        self.close_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data):
        """
        For loop which compiles time bars.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: Contains 3 columns - date_time, price, and volume.
        """

        # Iterate over rows
        list_bars = []

        for row in data.values:
            # Set variables
            date_time = row[0].timestamp()  # Convert to UTC timestamp
            self.tick_num += 1
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            timestamp_threshold = (int(
                float(date_time)) // self.threshold + 1) * self.threshold  # Current tick boundary timestamp

            # Init current bar timestamp with first ticks boundary timestamp
            if self.timestamp is None:
                self.timestamp = timestamp_threshold
            # Bar generation condition
            # Current ticks bar timestamp differs from current bars timestamp
            elif self.timestamp < timestamp_threshold:
                self._create_bars(self.timestamp, self.close_price,
                                  self.high_price, self.low_price, list_bars)

                # Reset cache
                self._reset_cache()
                self.timestamp = timestamp_threshold  # Current bar timestamp update

            # Update counters
            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Update close price
            self.close_price = price

            # Calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

        return list_bars


def get_time_bars(file_path_or_df, resolution='D', num_units=1, batch_size=20000000, verbose=True, to_csv=False,
                  output_path=None):
    """
    Creates Time Bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str or pd.DataFrame) Path to the csv file or Pandas Data Frame containing raw tick data in the format[date_time, price, volume]
    :param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')
    :param num_units: (int) Number of resolution units (3 days for example, 2 hours)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (int) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: Dataframe of time bars, if to_csv=True return None
    """

    bars = TimeBars(file_path_or_df=file_path_or_df, resolution=resolution, num_units=num_units, batch_size=batch_size)
    time_bars = bars.batch_run(verbose=verbose, to_csv=to_csv, output_path=output_path)
    return time_bars
