import pandas as pd
import numpy as np


class ETFTrick:
    def __init__(self, open_df, close_df, weights_df, costs_df, rates_df=None, in_memory=True):
        self.securities = self.close_df.columns()
        if rates_df is None:
            rates_df = open_df.copy()
            # set trivial(1.0) exchange rate if no data is provided
            rates_df[self.securities] = 1.0
        if not open_df.index.all() == close_df.index.all() == weights_df.index.all() == costs_df.index.all() == rates_df.index.all():
            raise ValueError('DataFrames indices are different')

        price_diff = close_df.diff()
        next_open_df = close_df.shift(-1)
        close_open_diff = close_df.substract(open_df)
        weights_df['abs_w_sum'] = weights_df.abs().sum(axis=1)
        delever_df = (next_open_df.mul(rates_df)).mul(
            weights_df['abs_w_sum'], axis=0)
        h_without_k = weights_df.div(delever_df)

        self.data_df = pd.concat([weights_df, h_without_k, close_open_diff, price_diff, costs_df, rates_df], axis=1, keys=[
                                 'w', 'h_t', 'close_open', 'price_diff', 'costs', 'rate'])
        self.data_df.dropna(inplace=True)
        self.in_memory = in_memory
        self.etf_series = pd.Series()  # etf series of
        self.prev_weights = np.array(
            [np.nan for i in range(0, len(self.securities))])  # init weights with nan values
        self.prev_h = None
        self.prev_k = 1.0  # init with 1$ as initial value
        self.prev_weights_change = False

    def get_etf_series(self):
        for row in self.data_df.values:
            index = row[0]
            weights_arr, h_t, close_open, price_diff, costs, rate = np.array_split(
                row[1:])
            weights_change = ~(prev_weights == weights_arr).all()
            if self.prev_weights_change is True:
                delta = close_open
            else:
                delta = price_diff
            if self.prev_h is None:
                self.prev_h = h_t * self.prev_k
                self.etf_series[index] = self.prev_k
                continue
            else:
                k = self.prev_k + \
                    np.sum(self.prev_h * rate * (delta + costs))
                self.etf_series[index] = k
                self.prev_k = k
                if weights_change is True:
                    self.prev_h = h_t * k
            self.prev_weights_change = weights_change
