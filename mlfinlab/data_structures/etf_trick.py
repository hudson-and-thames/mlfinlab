import pandas as pd
import numpy as np


class ETFTrick:
    def __init__(self, open_df, close_df, alloc_df, costs_df, rates_df=None, in_memory=True, batch_size=5000, index_col=0):
        self.in_memory = in_memory
        self.batch_size = batch_size
        self.index_col = index_col
        self.prev_k = 1.0  # init with 1$ as initial value
        self.prev_weights_change = False  # we need to track allocations vector change on previous step

        if in_memory is False:
            # create file iterators
            self.open_iter = pd.read_csv(open_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
            self.close_iter = pd.read_csv(close_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
            self.alloc_iter = pd.read_csv(alloc_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
            self.costs_iter = pd.read_csv(costs_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])

            open_df = self.open_iter.get_chunk(self.batch_size)
            close_df = self.close_iter.get_chunk(self.batch_size)
            alloc_df = self.alloc_iter.get_chunk(self.batch_size)
            costs_df = self.costs_iter.get_chunk(self.batch_size)

            if rates_df is not None:
                self.rates_iter = pd.read_csv(rates_df, iterator=True, index_col=self.index_col, parse_dates=[self.index_col])
                rates_df = self.rates_iter.get_chunk(batch_size)

        self.securities = alloc_df.columns # get all securities columns
        if rates_df is None:
            rates_df = open_df.copy()
            # set trivial(1.0) exchange rate if no data is provided
            rates_df[self.securities] = 1.0

        open_df = open_df[self.securities]  # align all securities columns in one order
        close_df = close_df[self.securities]
        alloc_df = alloc_df[self.securities]
        costs_df = costs_df[self.securities]
        rates_df = rates_df[self.securities]

        self.cache = {'open': open_df.iloc[-2:], 'close': close_df.iloc[-2:], 'alloc': alloc_df.iloc[-2:], 'costs': costs_df.iloc[-2:],
                      'rates': rates_df.iloc[-2:]}

        for df in [close_df, alloc_df, costs_df, rates_df]:
            if open_df.index.difference(df.index).shape[0] != 0:
                raise ValueError('DataFrames indices are different')

        self.data_df = self.generate_trick_components(open_df, close_df, alloc_df, costs_df, rates_df)
        self.data_df = self.data_df.iloc[1:]  # delete first nans (first row of close price difference is nan)
        self.prev_weights = np.array(
            [np.nan for _ in range(0, len(self.securities))])  # init weights with nan values

        self.prev_h = None # to find current etf_trick value we need previous h value

    def generate_trick_components(self, open_df, close_df, alloc_df, costs_df, rates_df):
        if self.in_memory is False:
            max_prev_index = self.cache['open'].index.max()
            second_max_prev_index = self.cache['open'].index[-2]
            open_df.loc[max_prev_index, :] = self.cache['open'].iloc[-1]
            close_df.loc[max_prev_index, :] = self.cache['close'].iloc[-1]
            alloc_df.loc[max_prev_index, :] = self.cache['alloc'].iloc[-1]
            costs_df.loc[max_prev_index, :] = self.cache['costs'].iloc[-1]
            rates_df.loc[max_prev_index, :] = self.cache['rates'].iloc[-1]

            close_df.loc[second_max_prev_index, :] = self.cache['close'].loc[second_max_prev_index, :]

            open_df.sort_index(inplace=True)
            close_df.sort_index(inplace=True)
            alloc_df.sort_index(inplace=True)
            costs_df.sort_index(inplace=True)
            rates_df.sort_index(inplace=True)

            open_df = open_df[self.securities]  # align all securities columns in one order
            close_df = close_df[self.securities]
            alloc_df = alloc_df[self.securities]
            costs_df = costs_df[self.securities]
            rates_df = rates_df[self.securities]

            price_diff = close_df.diff().iloc[1:]
            close_df = close_df.iloc[1:]

        else:
            price_diff = close_df.diff()

        next_open_df = open_df.shift(-1)
        close_open_diff = close_df.sub(open_df)  # close - open data frame
        alloc_df['abs_w_sum'] = alloc_df.abs().sum(
            axis=1)  # for each row generate absolute values sum for all assets
        delever_df = (next_open_df.mul(rates_df)).mul(
            alloc_df['abs_w_sum'], axis='index')  # deleverage component for h_i_t calculation
        h_without_k = alloc_df.div(
            delever_df)  # generate calculated h_t values for each row. For complete h_t calculation multiplying by current K_t is needed

        weights_df = alloc_df[self.securities]  # allign all securities columns
        h_without_k = h_without_k[self.securities]
        close_open_diff = close_open_diff[self.securities]
        price_diff = price_diff[self.securities]
        costs_df = costs_df[self.securities]
        rates_df = rates_df[self.securities]

        return pd.concat([weights_df, h_without_k, close_open_diff, price_diff, costs_df, rates_df], axis=1,
                                 keys=[
                                     'w', 'h_t', 'close_open', 'price_diff', 'costs',
                                     'rate'])  # generate data frame with all pregenerated info needed for ETF trick


    def _csv_file_etf_series(self):
        etf_series = pd.Series()
        while True:
            try:
                chunk_etf_series = self._chunk_loop()
                etf_series = etf_series.append(chunk_etf_series)
                open_df = self.open_iter.get_chunk(self.batch_size)
                close_df = self.close_iter.get_chunk(self.batch_size)
                alloc_df = self.alloc_iter.get_chunk(self.batch_size)
                costs_df = self.costs_iter.get_chunk(self.batch_size)

                try:
                    rates_df = self.rates_iter.get_chunk(self.batch_size)
                except AttributeError:
                    rates_df = open_df.copy()
                    # set trivial(1.0) exchange rate if no data is provided
                    rates_df[self.securities] = 1.0

                next_data_df = self.generate_trick_components(open_df, close_df, alloc_df, costs_df, rates_df)
                self.cache = {'open': open_df.iloc[-2:], 'close': close_df.iloc[-2:], 'alloc': alloc_df.iloc[-2:],
                              'costs': costs_df.iloc[-2:],
                              'rates': rates_df.iloc[-2:]}
                next_data_df.sort_index(inplace=True)
                self.data_df = next_data_df
                self.prev_k = etf_series.iloc[-2]

            except StopIteration:
                return etf_series

    def _chunk_loop(self):
        etf_series = pd.Series()
        for index, row in zip(self.data_df.index, self.data_df.values):
            weights_arr, h_t, close_open, price_diff, costs, rate = np.array_split(
                row, 6)
            weights_arr = np.nan_to_num(weights_arr)
            weights_change = bool(~(self.prev_weights == weights_arr).all())  # convert np.bool to bool
            if self.prev_weights_change is True:
                delta = close_open
            else:
                delta = price_diff
            if self.prev_h is None:
                self.prev_h = h_t * self.prev_k
                etf_series[index] = self.prev_k
            else:
                k = self.prev_k + \
                    np.nansum(self.prev_h * rate * (delta + costs))
                etf_series[index] = k
                self.prev_k = k
                if weights_change == True:
                    self.prev_h = h_t * k
                self.prev_weights_change = weights_change
                self.prev_weights = weights_arr

                if index == pd.Timestamp(2014, 5, 1, 3, 0, 53):
                    print(self.prev_k, np.nansum(self.prev_h * rate * (delta + costs)), np.nansum(delta))
        return etf_series


    def _in_memory_etf_series(self):
        return self._chunk_loop()

    def get_etf_series(self):
        if self.in_memory is True:
            return self._in_memory_etf_series()
        else:
            return self._csv_file_etf_series()

open_df = pd.read_csv('../open_data.csv', index_col=0)
close_df = pd.read_csv('../close_data.csv', index_col=0)
alloc_df = pd.read_csv('../alloc_data.csv', index_col=0)
costs_df = pd.read_csv('../costs_data.csv', index_col=0)
rates_df = pd.read_csv('../rates_data.csv', index_col=0)


open_df.index = pd.to_datetime(open_df.index)
close_df.index = pd.to_datetime(close_df.index)
alloc_df.index = pd.to_datetime(alloc_df.index)
costs_df.index = pd.to_datetime(costs_df.index)
rates_df.index = pd.to_datetime(rates_df.index)

open_df = '../open_data.csv'
close_df = '../close_data.csv'
alloc_df = '../alloc_data.csv'
costs_df = '../costs_data.csv'
rates_df = '../rates_data.csv'

test = ETFTrick(open_df, close_df, alloc_df,
                costs_df, rates_df, in_memory=False)
res = test.get_etf_series()
res.to_csv('res_csv.csv')