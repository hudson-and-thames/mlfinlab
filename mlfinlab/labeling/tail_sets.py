# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd


class TailSetLabels:
    """
    Tail set labels are a classification labeling technique introduced in the following paper: Nonlinear support vector
    machines can systematically identify stocks with high and low future returns. Algorithmic Finance, 2(1), pp.45-58.

    A tail set is defined to be a group of stocks whose volatility-adjusted return is in the highest or lowest
    quantile, for example the highest or lowest 5%.

    A classification model is then fit using these labels to determine which stocks to buy and sell in a long / short
    portfolio.
    """

    def __init__(self, prices, n_bins, vol_adj=None, window=None):
        """
        :param prices: (pd.DataFrame) Asset prices.
        :param n_bins: (int) Number of bins to determine the quantiles for defining the tail sets. The top and
                        bottom quantiles are considered to be the positive and negative tail sets, respectively.
        :param vol_adj: (str) Whether to take volatility adjusted returns. Allowable inputs are ``None``,
                        ``mean_abs_dev``, and ``stdev``.
        :param window: (int) Window period used in the calculation of the volatility adjusted returns, if vol_adj is not
                        None. Has no impact if vol_adj is None.
        """
        assert prices.shape[1] >= n_bins, "n_bins exceeds the number of stocks!"
        if vol_adj is not None:
            assert isinstance(window, int), "If vol_adj is not None, window must be int."
            assert len(prices) > window, "Length of price data must be greater than the window."

        self.prices = prices
        self.rets = np.log(prices).diff().dropna()
        self.window = window
        self.n_bins = n_bins
        self.vol_adj = vol_adj

        # Properties relating to the tail sets.
        self.vol_adj_rets = self.rets  # If no vol_adj.

        # Compute tail sets.
        if self.vol_adj is not None:
            self._vol_adjusted_rets()
        self.tail_sets = self.vol_adj_rets.dropna().apply(self._extract_tail_sets, axis=1)
        self.positive_tail_set = self.tail_sets.apply(self._positive_tail_set, axis=1)
        self.negative_tail_set = self.tail_sets.apply(self._negative_tail_set, axis=1)

    def get_tail_sets(self):
        """
        Computes the tail sets (positive and negative) and then returns a tuple with 3 elements, positive set, negative
        set, full matrix set.

        The positive and negative sets are each a series of lists with the names of the securities that fall within each
        set at a specific timestamp.

        For the full matrix a value of 1 indicates the volatility adjusted returns were in the top quantile, a value of
        -1 for the bottom quantile.
        :return: (tuple) positive set, negative set, full matrix set.
        """
        return self.positive_tail_set, self.negative_tail_set, self.tail_sets

    def _vol_adjusted_rets(self):
        """
        Computes the volatility adjusted returns. This is simply the log returns divided by a volatility estimate. We
        have provided 2 techniques for volatility estimation: an exponential moving average and the traditional standard
        deviation.
        """
        window = self.window

        # Have 2 measure of vol, the mean absolute, and stdev.
        if self.vol_adj == 'mean_abs_dev':
            # Huffman and Moll (2011) show that risk measured as the mean absolute deviation has more explanatory
            # power for future expected returns than standard deviation.
            vol = self.rets.abs().ewm(span=window, min_periods=window).mean()
        elif self.vol_adj == 'stdev':
            vol = self.rets.rolling(window).std()
        else:
            raise Exception('Invalid name for vol_adj. Valid names are ''mean_abs_dev'', ''stdev'', or None.')
        # Save vol adj rets.
        self.vol_adj_rets = (self.rets / vol).dropna()

    def _extract_tail_sets(self, row):
        """
        Method used in a .apply() setting to transform each row in a DataFrame to the positive and negative tail sets.

        This method splits the data into quantiles determined by the user, with n_bins.

        :param row: (pd.Series) Vol adjusted returns for a given date.
        :return: (pd.Series) Tail set with positive and negative labels.
        """
        # Get quantile labels.
        row = row.rank(method='first')  # To avoid error with unique bins when using qcut due to too many 0 values.
        row_quantiles = pd.qcut(x=row, q=self.n_bins, labels=range(1, 1 + self.n_bins), retbins=False)

        # Set class labels.
        row_quantiles = row_quantiles.to_numpy()  # Convert to numpy array
        row_quantiles[(row_quantiles != 1) & (row_quantiles != self.n_bins)] = 0
        row_quantiles[row_quantiles == 1] = -1
        row_quantiles[row_quantiles == self.n_bins] = 1

        # Convert to series.
        row_quantiles = pd.Series(row_quantiles, index=row.index)

        return row_quantiles

    @staticmethod
    def _positive_tail_set(row):
        """
        Takes as input a row from the vol_adj_ret DataFrame and then returns a list of names of the securities in the
        positive tail set, for this specific row date.

        This method is used in an apply() setting.

        :param row: (pd.Series) Labeled row of several stocks where each is already labeled with +1 (positive tail set),
                    -1 (negative tail set), or 0.
        :return: (list) Securities in the positive tail set.
        """
        return list(row[row == 1].index)

    @staticmethod
    def _negative_tail_set(row):
        """
        Takes as input a row from the vol_adj_ret DataFrame and then returns a list of names of the securities in the
        negative tail set, for this specific row date.

        This method is used in an apply() setting.

        :param row: (pd.Series) Labeled row of several stocks where each is already labeled with +1 (positive tail set),
                    -1 (negative tail set), or 0.
        :return: (list) Securities in the negative tail set.
        """
        return list(row[row == -1].index)
