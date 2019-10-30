from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy
from mlfinlab.microstructural_features.encoding import encode_array
from mlfinlab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap
import pandas as pd
import numpy as np


def _crop_data_frame_in_batches(df, chunksize):
    # pylint: disable=invalid-name
    """
    Splits df into chunks of chunksize
    :param df: (pd.DataFrame) to split
    :param chunksize: (Int) number of rows in chunk
    :return: (list) of chunks (pd.DataFrames)
    """
    generator_object = []
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        generator_object.append(chunk)
    return generator_object


class MicrostructuralFeaturesGenerator:
    """
    Abstract base class which contains the structure which is shared between the various standard and information
    driven bars. There are some methods contained in here that would only be applicable to information bars but
    they are included here so as to avoid a complicated nested class structure.
    """

    def __init__(self, trades_input, bar_index, batch_size=2e7, volume_encoding=None, pct_encoding=None):
        """
        Constructor
        :param trades_input: (String) Path to the csv file or Pandas Dat Frame containing raw tick data in the format[date_time, price, volume]
        :param metric: (String) type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (Int) Number of rows to read in from the csv, per batch.
        """

        if isinstance(trades_input, str):
            self.generator_object = pd.read_csv(trades_input, chunksize=batch_size, parse_dates=[0])
            # Read in the first row & assert format
            first_row = pd.read_csv(trades_input, nrows=1)
            self._assert_csv(first_row)
        elif isinstance(trades_input, pd.DataFrame):
            self.generator_object = _crop_data_frame_in_batches(trades_input, batch_size)
        else:
            raise ValueError('trades_input is neither string(path to a csv file) nor pd.DataFrame')

        # Base properties
        bar_index_iterator = iter(bar_index)
        self.current_date_time = bar_index_iterator.__next__()

        # Cache properties
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []

        # Entropy properties
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding

        # Batch_run properties
        self.flag = False  # The first flag is false since the first batch doesn't use the cache
        self.prev_price = None
        self.prev_tick_rule = 0

    def batch_run(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a csv file in batches and then constructs the financial data structure in the form of a DataFrame.
        The csv file must have only 3 columns: date_time, price, & volume.
        :param verbose: (Boolean) Flag whether to print message on each processed batch or not
        :param to_csv: (Boolean) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (Boolean) Path to results file, if to_csv = True
        :return: (DataFrame or None) Financial data structure
        """

        if to_csv is True:
            header = True  # if to_csv is True, header should be written on the first batch only
            open(output_path, 'w').close()  # clean output csv file

        if verbose:  # pragma: no cover
            print('Reading data in batches:')

        # Read csv in batches
        count = 0
        final_bars = []
        cols = ['date_time', 'avg_tick_size', 'tick_rule_sum', 'vwap', 'kyle_lambda', 'amihud_lambda',
                'hasbrouck_lambda', ]

        for en in ['shannon', 'plug_in', 'lempel_ziv']:
            cols += ['tick_rule_entropy' + en]

        if self.volume_encoding is not None:
            for en in ['shannon', 'plug_in', 'lempel_ziv']:
                cols += ['volume_entropy' + en]

        if self.pct_encoding is not None:
            for en in ['shannon', 'plug_in', 'lempel_ziv']:
                cols += ['price_entropy' + en]

        for batch in self.generator_object:
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
            print('Returning features \n')

        # Return a DataFrame
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df

        # Processed DataFrame is stored in .csv file, return None
        return None

    def _reset_cache(self):
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []

    def _extract_bars(self, data):
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.
        :param data: Contains 3 columns - date_time, price, and volume.
        """

        # Iterate over rows
        list_bars = []

        for row in data.values:
            # Set variables
            date_time = row[0]
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            # derivative variables
            price_diff = self._get_price_diff(price)
            log_ret = self._get_log_ret(price)

            self.price_diff.append(price_diff)
            self.trade_size.append(volume)
            self.tick_rule.append(signed_tick)
            self.dollar_size.append(dollar_value)
            self.log_ret.append(log_ret)

            self.prev_price = price

            # If date_time reached bar index
            if date_time >= self.current_date_time:
                self._get_bar_features(date_time, list_bars)

                # Take the next timestamp
                self.current_date_time.__next__()
                # Reset cache
                self._reset_cache()
        return list_bars

    def _get_bar_features(self, date_time, list_bars):
        features = [date_time]

        # Tick rule sum, avg tick size, VWAP
        features.append(sum(self.tick_rule))
        features.append(get_avg_tick_size(self.trade_size))
        features.append(vwap(self.dollar_size, self.trade_size))

        # Lambdas
        features.append(get_trades_based_kyle_lambda(self.price_diff, self.trade_size, self.tick_rule))  # Kyle lambda
        features.append(get_trades_based_amihud_lambda(self.log_ret, self.dollar_size))  # Amihud lambda
        features.append(
            get_trades_based_hasbrouck_lambda(self.log_ret, self.dollar_size, self.tick_rule))  # Hasbrouck lambda

        # Entropy features
        features.append(get_shannon_entropy(self.tick_rule))
        features.append(get_plug_in_entropy(self.tick_rule))
        features.append(get_lempel_ziv_entropy(self.tick_rule))

        if self.volume_encoding is not None:
            message = encode_array(self.trade_size, self.volume_encoding)
            features.append(get_shannon_entropy(message))
            features.append(get_plug_in_entropy(message))
            features.append(get_lempel_ziv_entropy(message))

        if self.pct_encoding is not None:
            message = encode_array(self.log_ret, self.volume_encoding)
            features.append(get_shannon_entropy(message))
            features.append(get_plug_in_entropy(message))
            features.append(get_lempel_ziv_entropy(message))

        list_bars.append(features)

    def _apply_tick_rule(self, price):
        """
        Applies the tick rule as defined on page 29.
        :param price: Price at time t
        :return: The signed tick
        """
        if self.prev_price is not None:
            tick_diff = price - self.prev_price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        return signed_tick

    def _get_price_diff(self, price):
        """
        """
        if self.prev_price is not None:
            return price - self.prev_price
        else:
            return 0  # First diff is assumed 0

    def _get_log_ret(self, price):
        """
        """
        if self.prev_price is not None:
            return np.log(price / self.prev_price)
        else:
            return 0  # First return is assumed 0

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
