"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features
"""

import pandas as pd
import numpy as np
from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy, \
    get_konto_entropy
from mlfinlab.microstructural_features.encoding import encode_array
from mlfinlab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap
from mlfinlab.microstructural_features.encoding import encode_tick_rule_array
from mlfinlab.util.misc import crop_data_frame_in_batches


# pylint: disable=too-many-instance-attributes

class MicrostructuralFeaturesGenerator:
    """
    Class which is used to generate inter-bar features when bars are already compressed.

    :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                               in the format[date_time, price, volume]
    :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
    :param batch_size: (int) Number of rows to read in from the csv, per batch.
    :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
    :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages

    """

    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None):
        """
        Constructor

        :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                                   in the format[date_time, price, volume]
        :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
        :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages
        """


        pass

    def get_features(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a csv file of ticks or pd.DataFrame in batches and then constructs corresponding microstructural intra-bar features:
        average tick size, tick rule sum, VWAP, Kyle lambda, Amihud lambda, Hasbrouck lambda, tick/volume/pct Shannon, Lempel-Ziv,
        Plug-in entropies if corresponding mapping dictionaries are provided (self.volume_encoding, self.pct_encoding).
        The csv file must have only 3 columns: date_time, price, & volume.

        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (bool) Path to results file, if to_csv = True
        :return: (DataFrame or None) Microstructural features for bar index
        """

        pass

    def _reset_cache(self):
        """
        Reset price_diff, trade_size, tick_rule, log_ret arrays to empty when bar is formed and features are
        calculated

        :return: None
        """

        pass

    def _extract_bars(self, data):
        """
        For loop which calculates features for formed bars using trades data

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        """

        pass

    def _get_bar_features(self, date_time: pd.Timestamp, list_bars: list) -> list:
        """
        Calculate inter-bar features: lambdas, entropies, avg_tick_size, vwap

        :param date_time: (pd.Timestamp) When bar was formed
        :param list_bars: (list) Previously formed bars
        :return: (list) Inter-bar features
        """

        pass

    def _apply_tick_rule(self, price: float) -> int:
        """
        Advances in Financial Machine Learning, page 29.

        Applies the tick rule

        :param price: (float) Price at time t
        :return: (int) The signed tick
        """

        pass

    def _get_price_diff(self, price: float) -> float:
        """
        Get price difference between ticks

        :param price: (float) Price at time t
        :return: (float) Price difference
        """

        pass

    def _get_log_ret(self, price: float) -> float:
        """
        Get log return between ticks

        :param price: (float) Price at time t
        :return: (float) Log return
        """

        pass

    @staticmethod
    def _assert_csv(test_batch):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) the first row of the dataset.
        :return: (None)
        """

        pass
