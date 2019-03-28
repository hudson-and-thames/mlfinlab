import pandas as pd
import numpy as np


class ETFTrick:
    def __init__(self, open_df, close_df, weights_df, costs_df, rates_df=None, in_memory=True):
        self.securities = weights_df.columns
        if rates_df is None:
            rates_df = open_df.copy()
            # set trivial(1.0) exchange rate if no data is provided
            rates_df[self.securities] = 1.0

        open_df = open_df[self.securities]
        close_df = close_df[self.securities]
        weights_df = weights_df[self.securities]
        costs_df = costs_df[self.securities]
        rates_df = rates_df[self.securities]

        for df in [close_df, weights_df, costs_df, rates_df]:
            if open_df.index.difference(df.index).shape[0] != 0:
                raise ValueError('DataFrames indices are different')

        price_diff = close_df.diff()
        next_open_df = open_df.shift(-1)
        close_open_diff = close_df.sub(open_df)
        weights_df['abs_w_sum'] = weights_df.abs().sum(axis=1)
        delever_df = (next_open_df.mul(rates_df)).mul(
            weights_df['abs_w_sum'], axis='index')
        h_without_k = weights_df.div(delever_df)


        weights_df = weights_df[self.securities]
        h_without_k = h_without_k[self.securities]
        close_open_diff = close_open_diff[self.securities]
        price_diff = price_diff[self.securities]
        costs_df = costs_df[self.securities]
        rates_df = rates_df[self.securities]

        self.data_df = pd.concat([weights_df, h_without_k, close_open_diff, price_diff, costs_df, rates_df], axis=1, keys=[
                                 'w', 'h_t', 'close_open', 'price_diff', 'costs', 'rate'])
        self.data_df = self.data_df.iloc[1:]  # delete first nans
        self.in_memory = in_memory
        self.prev_weights = np.array(
            [np.nan for i in range(0, len(self.securities))])  # init weights with nan values
        self.prev_h = None
        self.prev_k = 1.0  # init with 1$ as initial value
        self.prev_weights_change = False

    def get_etf_series(self):
        etf_series = pd.Series()
        for index, row in zip(self.data_df.index, self.data_df.values):
            weights_arr, h_t, close_open, price_diff, costs, rate = np.array_split(
                row, 6)
            weights_arr = np.nan_to_num(weights_arr)
            weights_change = bool(~(self.prev_weights == weights_arr).all()) # convert np.bool to bool
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
        return etf_series

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

test = ETFTrick(open_df, close_df, alloc_df,
                costs_df, rates_df, in_memory=True)
res = test.get_etf_series()
res.to_csv('res.csv')