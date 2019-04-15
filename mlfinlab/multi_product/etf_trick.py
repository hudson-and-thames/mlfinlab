"""
This module contains class for ETF trick generation and futures roll function, described in
Marcos Lopez de Prado book 'Advances in Financial Machine Learning'
ETF trick class can generate ETF trick series either from .csv files or from in memory pandas
data frames
"""

# Imports
import pandas as pd
import numpy as np


class ETFTrick(object):
    """
    Contains logic of vectorised ETF trick implementaion. Can used for both memory data frames (pd.DataFrame) and csv files.
    All data frames, files should be processed in a specific format, described in examples
    """

    def __init__(self, open_df, close_df, alloc_df, costs_df, rates_df=None, in_memory=True, batch_size=5000,
                 index_col=0):
        """
        Constructor
        Creates class object, for csv based files reads the first data chunk.
        :param open_df: (pd.DataFrame or string): open prices data frame or path to csv file, corresponds to o(t) from the book
        :param close_df: (pd.DataFrame or string): close prices data frame or path to csv file, corresponds to p(t)
        :param alloc_df: (pd.DataFrame or string): asset allocations data frame or path to csv file (in # of contracts), corresponds to w(t)
        :param costs_df: (pd.DataFrame or string): rebalance, carry and dividend costs of holding/rebalancing the position, corresponds to d(t)
        :param rates_df: (pd.DataFrame or string): dollar value of one point move of contract
                                                   includes exchange rate, futures contracts multiplies). Corresponds to phi(t)
                         For example, 1$ in VIX index, equals 1000$ in VIX futures contract value. If None then trivial (all values equal 1.0) is generated
        :param in_memory: (boolean): boolean flag of whether data frames are stored in memory or in csv files
        :param batch_size: (int): number of rows to per batch. Used for csv stored data frames
        :param index_col: (int): positional index of index column. Used for to determine index column in csv files

        """
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.index_col = index_col
        self.prev_k = 1.0  # init with 1$ as initial value
        # we need to track allocations vector change on previous step
        # previous allocation change is needed for delta component calculation
        self.prev_allocs_change = False
        self.data_dict = dict.fromkeys(['open', 'close', 'alloc', 'costs', 'rates'], None)

        if in_memory is False:
            self.iter_dict = dict.fromkeys(['open', 'close', 'alloc', 'costs', 'rates'], None)
            # create file iterators
            self.iter_dict['open'] = pd.read_csv(
                open_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
            self.iter_dict['close'] = pd.read_csv(
                close_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
            self.iter_dict['alloc'] = pd.read_csv(
                alloc_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
            self.iter_dict['costs'] = pd.read_csv(
                costs_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])

            # get initial data chunk from file iterator
            self.data_dict['open'] = self.iter_dict['open'].get_chunk(self.batch_size)
            self.data_dict['close'] = self.iter_dict['close'].get_chunk(self.batch_size)
            self.data_dict['alloc'] = self.iter_dict['alloc'].get_chunk(self.batch_size)
            self.data_dict['costs'] = self.iter_dict['costs'].get_chunk(self.batch_size)

            if rates_df is not None:
                self.iter_dict['rates'] = pd.read_csv(
                    rates_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
                self.data_dict['rates'] = self.iter_dict['rates'].get_chunk(batch_size)

        self.securities = self.data_dict['alloc'].columns  # get all securities columns

        if rates_df is None:
            self.data_dict['rates'] = open_df.copy()
            # set trivial(1.0) exchange rate if no data is provided
            self.data_dict['rates'][self.securities] = 1.0

        # align all securities columns in one order
        for df_name in self.data_dict.keys():
            self.data_dict[df_name] = self.data_dict[df_name][self.securities]

        # cache is used to recalculate previous row after next chunk load
        self.cache = self._update_cache()

        # check if all data frames have the same index
        for temp_df in self.data_dict.values():
            if self.data_dict['open'].index.difference(temp_df.index).shape[0] != 0:
                raise ValueError('DataFrames indices are different')

        self.data_df = self.generate_trick_components()  # get all possible etf trick calculations which can be vectorised
        # delete first nans (first row of close price difference is nan)
        self.data_df = self.data_df.iloc[1:]
        self.prev_allocs = np.array(
            [np.nan for _ in range(0, len(self.securities))])  # init weights with nan values

        self.prev_h = None  # to find current etf_trick value we need previous h value

    def generate_trick_components(self):
        """
        Calculates all etf trick operations which can be vectorised. Outputs multilevel pandas data frame.
        Generated components:
        'w': alloc_df
        'h_t': h_t/K value from ETF trick algorithm from the book. Which K to use is based on previous values and cannot be vectorised
        'close_open': close_df - open_df
        'price_diff': close price differences
        'costs': costs_df
        'rate': rates_df
        :return: (pd.DataFrame): pandas data frame with columns in a format: component_1/asset_name_1, component_1/asset_name_2, ..., component_6/asset_name_n
        """
        if self.in_memory is False:
            # latest index from previous data_df(cache)
            max_prev_index = self.cache['open'].index.max()
            second_max_prev_index = self.cache['open'].index[-2]
            # add the last row from previous data chunk to a new chunk
            for df_name in self.data_dict.keys():
                temp_df = self.data_dict[df_name]
                temp_df.loc[max_prev_index, :] = self.cache[df_name].iloc[-1]
                self.data_dict[df_name] = temp_df

            # to recalculate latest row we need close price differences
            self.data_dict['close'].loc[second_max_prev_index, :] = self.cache['close'].loc[second_max_prev_index, :]
            # that is why close_df needs 2 previous chunk rows to omit first row nans

            for df_name in self.data_dict.keys():
                self.data_dict[df_name].sort_index(inplace=True)  # sort data frames after all appends
                self.data_dict[df_name] = self.data_dict[df_name][
                    self.securities]  # align all securities columns in one order

            # get price diffs, take values from the second row (first is nan)
            price_diff = self.data_dict['close'].diff().iloc[1:]
            # delete second max row from previous data chunk in close_df
            self.data_dict['close'] = self.data_dict['close'].iloc[1:]
        else:
            price_diff = self.data_dict['close'].diff()

        next_open_df = self.data_dict['open'].shift(-1)  # generate next open prices
        close_open_diff = self.data_dict['close'].sub(self.data_dict['open'])  # close - open data frame
        self.data_dict['alloc']['abs_w_sum'] = self.data_dict['alloc'].abs().sum(
            axis=1)  # for each row generate absolute values sum for all assets

        # allocations deleverage component
        delever_df = self.data_dict['alloc'].div(self.data_dict['alloc']['abs_w_sum'], axis='index')
        next_open_mul_rates_df = next_open_df.mul(
            self.data_dict['rates'], axis='index')  # o(t+1) * phi(t)

        # generate calculated h_t values for each row
        # for complete h_t calculation multiplying by current K_t is needed(can't be vectorised)
        h_without_k = delever_df.div(next_open_mul_rates_df)

        weights_df = self.data_dict['alloc'][self.securities]  # allign all securities columns
        h_without_k = h_without_k[self.securities]
        close_open_diff = close_open_diff[self.securities]
        price_diff = price_diff[self.securities]

        return pd.concat([weights_df, h_without_k, close_open_diff, price_diff, self.costs_df, self.rates_df], axis=1,
                         keys=[
                             'w', 'h_t', 'close_open', 'price_diff', 'costs',
                             'rate'])  # generate data frame with all pregenerated info needed for ETF trick

    def _update_cache(self):
        cache_dict = {'open': self.data_dict['open'].iloc[-2:], 'close': self.data_dict['close'].iloc[-2:],
                      'alloc': self.data_dict['alloc'].iloc[-2:], 'costs': self.data_dict['costs'].iloc[-2:],
                      'rates': self.data_dict['rates'].iloc[-2:]}
        return cache_dict

    def _chunk_loop(self):
        """
        Single ETF trick iteration for currently stored data set in memory.
        For in-memory data set would yield complete ETF trick series, for csv based
        would generate ETF trick series for current batch.
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        etf_series = pd.Series()
        for index, row in zip(self.data_df.index, self.data_df.values):
            weights_arr, h_t, close_open, price_diff, costs, rate = np.array_split(
                row, 6)  # split row in corresponding values for ETF trick
            # replaces nan to zeros in allocations vector
            weights_arr = np.nan_to_num(weights_arr)

            # convert np.bool to bool
            # boolean flag of allocations vector change
            allocs_change = bool(
                ~(self.prev_allocs == weights_arr).all())  # not(all elements in prev_w equal current_w)
            if self.prev_allocs_change is True:
                delta = close_open  # delta from book algorithm
            else:
                delta = price_diff
            if self.prev_h is None:
                # previous h value is needed for current K calculation. The first iteration sets up prev_h value
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
                # update previous allocation vector change
                self.prev_allocs_change = allocs_change
                self.prev_allocs = weights_arr
        return etf_series

    def _csv_file_etf_series(self):
        """
        Csv based ETF trick series generation
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        etf_series = pd.Series()
        # read data in batch until StopIteration exception is raised
        while True:
            try:
                # generate ETF trick values for read data chunk
                chunk_etf_series = self._chunk_loop()
                etf_series = etf_series.append(chunk_etf_series)

                # read the next batch
                self.data_dict['open'] = self.iter_dict['open'].get_chunk(self.batch_size)
                self.data_dict['close'] = self.iter_dict['close'].get_chunk(self.batch_size)
                self.data_dict['alloc'] = self.iter_dict['alloc'].get_chunk(self.batch_size)
                self.data_dict['costs'] = self.iter_dict['costs'].get_chunk(self.batch_size)

                if self.iter_dict['rates'] is not None:
                    # if there is rates_df iterator, get the next chunk
                    self.data_dict['rates'] = self.data_dict['rates'].get_chunk(self.batch_size)
                else:
                    # if no iterator is available, generate trivial ratest_df (1.0 for all securities)
                    self.data_dict['rates'] = self.data_dict['open'].copy()
                    # set trivial(1.0) exchange rate if no data is provided
                    self.data_dict['rates'][self.securities] = 1.0

                next_data_df = self.generate_trick_components()
                self.cache = self._update_cache()
                next_data_df.sort_index(inplace=True)
                self.data_df = next_data_df  # update data_df for ETF trick calculation
                # reset prev_k for previous row calculation
                self.prev_k = etf_series.iloc[-2]
            except StopIteration:
                return etf_series

    def _in_memory_etf_series(self):
        """
        In-memory based ETF trick series generation
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        return self._chunk_loop()

    def get_etf_series(self):
        """
        External method which defines which etf trick method to use based on in_memory field value
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """
        if self.in_memory is True:
            etf_trick_series = self._in_memory_etf_series()
        else:
            etf_trick_series = self._csv_file_etf_series()
        return etf_trick_series


def get_futures_roll_series(data_df, open_col, close_col, sec_col, current_sec_col, roll_backward=False):
    """
    Function for generating rolling futures series from data frame of multiple futures
    :param data_df: (pd.DataFrame): pandas DataFrame containing price info, security name and  current active futures column
    :param open_col: (string): open prices column name
    :param close_col: (string): close prices column name
    :param sec_col: (string): security name column name
    :param current_sec_col: (string): current active security column name.
                                      When value in this column changes it means rolling
    :param roll_backward: (boolean): True for substracting final gap value from all values
    :return (pd.Series): futures roll close price series
    """
    # filter out security data which is not used as current security
    filtered_df = data_df[data_df[sec_col] == data_df[current_sec_col]]
    filtered_df.sort_index(inplace=True)
    # generate roll dates series based on curren_sec column value change
    roll_dates = filtered_df[current_sec_col].drop_duplicates(keep='first').index
    gaps = filtered_df[close_col] * 0  # roll gaps series
    # on roll dates, gap equals open - close
    gaps.loc[roll_dates[1:]] = filtered_df[open_col].loc[roll_dates[1:]] - filtered_df[close_col].loc[
        roll_dates[1:]]  # TODO: understand why Marcos used iloc and list logic
    if roll_backward:
        gaps -= gaps.iloc[-1]  # roll backward
    data_df[close_col] -= gaps
    return data_df[close_col]
