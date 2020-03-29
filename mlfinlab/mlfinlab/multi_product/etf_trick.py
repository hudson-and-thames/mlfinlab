"""
This module contains class for ETF trick generation and futures roll function, described in Marcos Lopez de Prado's
book 'Advances in Financial Machine Learning' ETF trick class can generate ETF trick series either from .csv files
or from in memory pandas DataFrames
"""

import warnings
import pandas as pd
import numpy as np


class ETFTrick:
    """
    Contains logic of vectorised ETF trick implementation. Can used for both memory data frames (pd.DataFrame) and
    csv files. All data frames, files should be processed in a specific format, described in examples
    """

    def __init__(self, open_df, close_df, alloc_df, costs_df, rates_df=None, index_col=0):
        """
        Constructor

        Creates class object, for csv based files reads the first data chunk.

        :param open_df: (pd.DataFrame or string): open prices data frame or path to csv file,
         corresponds to o(t) from the book
        :param close_df: (pd.DataFrame or string): close prices data frame or path to csv file, corresponds to p(t)
        :param alloc_df: (pd.DataFrame or string): asset allocations data frame or path to csv file (in # of contracts),
         corresponds to w(t)
        :param costs_df: (pd.DataFrame or string): rebalance, carry and dividend costs of holding/rebalancing the
         position, corresponds to d(t)
        :param rates_df: (pd.DataFrame or string): dollar value of one point move of contract includes exchange rate,
         futures contracts multiplies). Corresponds to phi(t)
         For example, 1$ in VIX index, equals 1000$ in VIX futures contract value.
         If None then trivial (all values equal 1.0) is generated
        :param index_col: (int): positional index of index column. Used for to determine index column in csv files
        """

        warnings.warn("This a beta version of ETF trick. Please proof check the results", DeprecationWarning)
        self.index_col = index_col
        self.prev_k = 1.0  # Init with $1 as initial value

        # We need to track allocations vector change on previous step
        # Previous allocation change is needed for delta component calculation
        self.prev_allocs_change = False
        self.prev_h = None  # To find current etf_trick value we need previous h value

        self.data_dict = {}
        self.iter_dict = None  # Dictionary of csv files iterators, None for in_memory ETF trick calculation
        self.init_fields = None  # Dictionary of initial fields values, needed to call reset method

        if isinstance(alloc_df, str):
            # String values for open, close, alloc, costs and rates mean that we generate ETF trick from csv files
            # Remember constructor fields for possible reset() method call
            self.init_fields = {'open_df': open_df, 'close_df': close_df, 'alloc_df': alloc_df, 'costs_df': costs_df,
                                'rates_df': rates_df, 'index_col': index_col}
            self.iter_dict = dict.fromkeys(['open', 'close', 'alloc', 'costs', 'rates'], None)

            # Create file iterators
            self.iter_dict['open'] = pd.read_csv(open_df,
                                                 iterator=True,
                                                 index_col=self.index_col,
                                                 parse_dates=[self.index_col])
            self.iter_dict['close'] = pd.read_csv(close_df,
                                                  iterator=True,
                                                  index_col=self.index_col,
                                                  parse_dates=[self.index_col])
            self.iter_dict['alloc'] = pd.read_csv(alloc_df,
                                                  iterator=True,
                                                  index_col=self.index_col,
                                                  parse_dates=[self.index_col])
            self.iter_dict['costs'] = pd.read_csv(costs_df,
                                                  iterator=True,
                                                  index_col=self.index_col,
                                                  parse_dates=[self.index_col])

            if rates_df is not None:
                self.iter_dict['rates'] = pd.read_csv(rates_df,
                                                      iterator=True,
                                                      index_col=self.index_col,
                                                      parse_dates=[self.index_col])

            # Get headers(column names) from csv files (except index col) which correspond to security names
            self.securities = list(pd.read_csv(alloc_df, nrows=0, header=0, index_col=self.index_col))

        elif isinstance(alloc_df, pd.DataFrame):
            self.data_dict['open'] = open_df
            self.data_dict['close'] = close_df
            self.data_dict['alloc'] = alloc_df
            self.data_dict['costs'] = costs_df
            self.data_dict['rates'] = rates_df
            self.securities = self.data_dict['alloc'].columns  # Get all securities columns

            if rates_df is None:
                self.data_dict['rates'] = open_df.copy()
                # Set trivial(1.0) exchange rate if no data is provided
                self.data_dict['rates'][self.securities] = 1.0

            # Align all securities columns in one order
            for df_name in self.data_dict:
                self.data_dict[df_name] = self.data_dict[df_name][self.securities]

            self._index_check()
        else:
            raise TypeError('Wrong input to ETFTrick class. Either strings with paths to csv files, or pd.DataFrames')

        self.prev_allocs = np.array([np.nan for _ in range(0, len(self.securities))])  # Init weights with nan values

    def _append_previous_rows(self, cache):
        """
        Uses latest two rows from cache to append into current data. Used for csv based ETF trick, when the next
        batch is loaded and we need to recalculate K value which corresponds to previous batch.

        :param cache: (dict): dictionary which pd.DataFrames with latest 2 rows of open, close, alloc, costs, rates
        :return: (pd.DataFrame): data frame with close price differences (updates self.data_dict)
        """
        # Latest index from previous data_df(cache)
        max_prev_index = cache['open'].index.max()
        second_max_prev_index = cache['open'].index[-2]
        # Add the last row from previous data chunk to a new chunk
        for df_name in self.data_dict:
            temp_df = self.data_dict[df_name]
            temp_df.loc[max_prev_index, :] = cache[df_name].iloc[-1]
            self.data_dict[df_name] = temp_df

        # To recalculate latest row we need close price differences
        # That is why close_df needs 2 previous chunk rows to omit first row nans
        self.data_dict['close'].loc[second_max_prev_index, :] = cache['close'].loc[second_max_prev_index, :]


        for df_name in self.data_dict:
            self.data_dict[df_name].sort_index(inplace=True)  # Sort data frames after all appends
            self.data_dict[df_name] = self.data_dict[df_name][
                self.securities]  # Align all securities columns in one order

        # Get price diffs, take values from the second row (first is nan)
        price_diff = self.data_dict['close'].diff().iloc[1:]
        # Delete second max row from previous data chunk in close_df
        self.data_dict['close'] = self.data_dict['close'].iloc[1:]

        return price_diff

    def generate_trick_components(self, cache=None):
        """
        Calculates all etf trick operations which can be vectorised. Outputs multilevel pandas data frame.

        Generated components:
        'w': alloc_df
        'h_t': h_t/K value from ETF trick algorithm from the book. Which K to use is based on previous values and
            cannot be vectorised.
        'close_open': close_df - open_df
        'price_diff': close price differences
        'costs': costs_df
        'rate': rates_df

        :param cache: (dict of pd.DataFrames): dictionary which contains latest 2 rows of open, close, rates, alloc,
            costs, rates data
        :return: (pd.DataFrame): pandas data frame with columns in a format: component_1/asset_name_1,
            component_1/asset_name_2, ..., component_6/asset_name_n
        """
        if cache:
            price_diff = self._append_previous_rows(cache)
        else:
            price_diff = self.data_dict['close'].diff()

        next_open_df = self.data_dict['open'].shift(-1)  # Generate next open prices
        close_open_diff = self.data_dict['close'].sub(self.data_dict['open'])  # close - open data frame
        # For each row generate absolute values sum for all assets
        self.data_dict['alloc']['abs_w_sum'] = self.data_dict['alloc'].abs().sum(axis=1)

        # Allocations de-leverage component
        delever_df = self.data_dict['alloc'].div(self.data_dict['alloc']['abs_w_sum'], axis='index')
        next_open_mul_rates_df = next_open_df.mul(self.data_dict['rates'], axis='index')  # o(t+1) * phi(t)

        # Generate calculated h_t values for each row
        # For complete h_t calculation multiplying by current K_t is needed(can't be vectorised)
        h_without_k = delever_df.div(next_open_mul_rates_df)

        weights_df = self.data_dict['alloc'][self.securities]  # Align all securities columns
        h_without_k = h_without_k[self.securities]
        close_open_diff = close_open_diff[self.securities]
        price_diff = price_diff[self.securities]

        # Generate data frame with all precomputed info needed for ETF trick
        final_df = pd.concat([weights_df, h_without_k, close_open_diff, price_diff,
                              self.data_dict['costs'], self.data_dict['rates']],
                             axis=1,
                             keys=['w', 'h_t', 'close_open', 'price_diff', 'costs', 'rate'])

        return final_df

    def _update_cache(self):
        """
        Updates cache (two previous rows) when new data batch is read into the memory. Cache is used to
        recalculate ETF trick value which corresponds to previous batch last row. That is why we need 2 previous rows
        for close price difference calculation

        :return: (dict): dictionary with open, close, alloc, costs and rates last 2 rows
        """
        cache_dict = {'open': self.data_dict['open'].iloc[-2:], 'close': self.data_dict['close'].iloc[-2:],
                      'alloc': self.data_dict['alloc'].iloc[-2:], 'costs': self.data_dict['costs'].iloc[-2:],
                      'rates': self.data_dict['rates'].iloc[-2:]}
        return cache_dict

    def _chunk_loop(self, data_df):
        """
        Single ETF trick iteration for currently stored(with needed components) data set in memory (data_df).
        For in-memory data set would yield complete ETF trick series, for csv based
        would generate ETF trick series for current batch.

        :param data_df: The data set on which to apply the ETF trick.
        :return: (pd.Series): pandas Series with ETF trick values
        """
        etf_series = pd.Series()
        for index, row in zip(data_df.index, data_df.values):
            # Split row in corresponding values for ETF trick
            # pylint: disable=unbalanced-tuple-unpacking
            weights_arr, h_t, close_open, price_diff, costs, rate = np.array_split(row, 6)
            # Replaces nan to zeros in allocations vector
            weights_arr = np.nan_to_num(weights_arr)

            # Convert np.bool to bool. Boolean flag of allocations vector change
            # Not(all elements in prev_w equal current_w)
            allocs_change = bool(~(self.prev_allocs == weights_arr).all())
            if self.prev_allocs_change is True:
                delta = close_open  # delta from book algorithm
            else:
                delta = price_diff
            if self.prev_h is None:
                # Previous h value is needed for current K calculation. The first iteration sets up prev_h value
                self.prev_h = h_t * self.prev_k
                # K is equal to 1 on the first iteration
                etf_series[index] = self.prev_k
            else:
                if self.prev_allocs_change is True:
                    # h_t is equal to previous h_t, or h_t * k if allocations vector changes (roll dates)
                    self.prev_h = h_t * self.prev_k

                k = self.prev_k + \
                    np.nansum(self.prev_h * rate * (delta + costs))
                etf_series[index] = k

                self.prev_k = k
                # Update previous allocation vector change
                self.prev_allocs_change = allocs_change
                self.prev_allocs = weights_arr

        return etf_series

    def _index_check(self):
        """
        Internal check for all price, rates and allocations data frames have the same index
        """
        # Check if all data frames have the same index
        for temp_df in self.data_dict.values():
            if self.data_dict['open'].index.difference(temp_df.index).shape[0] != 0 or \
                    self.data_dict['open'].shape != temp_df.shape:
                raise ValueError('DataFrames indices are different')

    def _get_batch_from_csv(self, batch_size):
        """
        Reads the next batch of data sets from csv files and puts them in class variable data_dict

        :param batch_size: number of rows to read
        """
        # Read the next batch
        self.data_dict['open'] = self.iter_dict['open'].get_chunk(batch_size)
        self.data_dict['close'] = self.iter_dict['close'].get_chunk(batch_size)
        self.data_dict['alloc'] = self.iter_dict['alloc'].get_chunk(batch_size)
        self.data_dict['costs'] = self.iter_dict['costs'].get_chunk(batch_size)

        if self.iter_dict['rates'] is not None:
            # If there is rates_df iterator, get the next chunk
            self.data_dict['rates'] = self.iter_dict['rates'].get_chunk(batch_size)
        else:
            # If no iterator is available, generate trivial rates_df (1.0 for all securities)
            self.data_dict['rates'] = self.data_dict['open'].copy()
            # Set trivial(1.0) exchange rate if no data is provided
            self.data_dict['rates'][self.securities] = 1.0

        # Align all securities columns in one order
        for df_name in self.data_dict:
            self.data_dict[df_name] = self.data_dict[df_name][self.securities]

        self._index_check()

    def _rewind_etf_trick(self, alloc_df, etf_series):
        """
        ETF trick uses next open price information, when we process csv file in batches the last row in batch will have
        next open price value as nan, that is why when new batch comes, we need to rewind ETF trick values one step
        back, recalculate ETF trick value for the last row from previous batch using open price from latest batch
        received. This function rewinds values needed for ETF trick calculation recalculate

        :param alloc_df: (pd.DataFrame): data frame with allocations vectors
        :param etf_series (pd.Series): current computed ETF trick series
        """
        self.prev_k = etf_series.iloc[-2]  # Reset prev_k for previous row calculation
        self.prev_allocs = alloc_df.iloc[-2]  # Reset previous allocations vector
        self.prev_allocs_change = bool(~(self.prev_allocs == alloc_df.iloc[-3]).all())

    def _csv_file_etf_series(self, batch_size):
        """
        Csv based ETF trick series generation

        :param: batch_size: (int): Size of the batch that you would like to make use of
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        etf_series = pd.Series()
        self._get_batch_from_csv(batch_size)
        # Data frame which contains all precomputed info for etf trick
        data_df = self.generate_trick_components(cache=None)  # Cache is empty on the first batch run
        cache = self._update_cache()
        # Delete first nans (first row of close price difference is nan)
        data_df = data_df.iloc[1:]
        # Drop last row value from previous batch (this row needs to be recalculated using new data)
        omit_last_row = False

        # Read data in batch until StopIteration exception is raised
        while True:
            try:
                chunk_etf_series = self._chunk_loop(data_df)
                if omit_last_row is True:
                    etf_series = etf_series.iloc[:-1]  # Delete last row (chunk_etf_series stores updated row value)
                etf_series = etf_series.append(chunk_etf_series)
                self._get_batch_from_csv(batch_size)
                self._rewind_etf_trick(data_df['w'], etf_series)  # Rewind etf series one step back
                data_df = self.generate_trick_components(cache)  # Update data_df for ETF trick calculation
                cache = self._update_cache()  # Update cache
                omit_last_row = True
            except StopIteration:
                return etf_series

    def _in_memory_etf_series(self):
        """
        In-memory based ETF trick series generation.

        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        data_df = self.generate_trick_components()  # Data frame which contains all precomputed info for etf trick

        # Delete first nans (first row of close price difference is nan)
        data_df = data_df.iloc[1:]
        return self._chunk_loop(data_df)

    def get_etf_series(self, batch_size=1e5):
        """
        External method which defines which etf trick method to use.

        :param: batch_size: Size of the batch that you would like to make use of
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        if self.iter_dict is None:
            etf_trick_series = self._in_memory_etf_series()
        else:
            if batch_size < 3:
                # We use latest 2 elements from prev batch, so minimum batch is 3
                raise ValueError('Batch size should be >= 3')
            etf_trick_series = self._csv_file_etf_series(batch_size)
        return etf_trick_series

    def reset(self):
        """
        Re-inits class object. This methods can be used to reset file iterators for multiple get_etf_trick() calls.
        """
        self.__init__(**self.init_fields)


def get_futures_roll_series(data_df, open_col, close_col, sec_col, current_sec_col, roll_backward=False,
                            method='absolute'):
    """
    Function for generating rolling futures series from data frame of multiple futures.

    :param data_df: (pd.DataFrame): pandas DataFrame containing price info, security name and current active futures
     column
    :param open_col: (string): open prices column name
    :param close_col: (string): close prices column name
    :param sec_col: (string): security name column name
    :param current_sec_col: (string): current active security column name. When value in this column changes it means
     rolling
    :param roll_backward: (boolean): True for subtracting final gap value from all values
    :param method: (string): what returns user wants to preserve, 'absolute' or 'relative'
    :return (pd.Series): futures roll close price series
    """

    # Filter out security data which is not used as current security
    filtered_df = data_df[data_df[sec_col] == data_df[current_sec_col]]
    filtered_df.sort_index(inplace=True)

    # Generate roll dates series based on current_sec column value change
    roll_dates = filtered_df[current_sec_col].drop_duplicates(keep='first').index
    timestamps = list(filtered_df.index)  # List of timestamps
    prev_roll_dates_index = [timestamps.index(i) - 1 for i in roll_dates]  # Dates before rolling date index (int)

    # On roll dates, gap equals open - close or open/close
    if method == 'absolute':
        gaps = filtered_df[close_col] * 0  # roll gaps series
        gaps.loc[roll_dates[1:]] = filtered_df[open_col].loc[roll_dates[1:]] - filtered_df[close_col].iloc[
            prev_roll_dates_index[1:]].values
        gaps = gaps.cumsum()

        if roll_backward:
            gaps -= gaps.iloc[-1]  # Roll backward diff
    elif method == 'relative':
        gaps = filtered_df[close_col] * 0 + 1  # Roll gaps series
        gaps.loc[roll_dates[1:]] = filtered_df[open_col].loc[roll_dates[1:]] / filtered_df[close_col].iloc[
            prev_roll_dates_index[1:]].values
        gaps = gaps.cumprod()

        if roll_backward:
            gaps /= gaps.iloc[-1]  # Roll backward div
    else:
        raise ValueError('The method must be either absolute or relative, Check spelling.')

    return gaps
