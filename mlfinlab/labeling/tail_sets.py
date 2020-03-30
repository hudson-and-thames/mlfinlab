# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd


class TailSetLabels:
    """
    Tail set labels are a classification labeling technique introduced in the following paper: Nonlinear support vector
    machines can systematically identify stocks with high and low future returns. Algorithmic Finance, 2(1), pp.45-58.

    A tail set is defined to be a group of stocks whose volatility-adjusted price change is in the highest or lowest
    quantile, for example the highest or lowest 5%.

    A classification model is then fit using these labels to determine which stocks to buy and sell in a long / short
    portfolio.
    """

    def __init__(self, prices, window, mean_abs_dev):
        """
        :param prices: (DataFrame) Asset prices.
        :param window: (int) Window period used in the calculation of the volatility.
        :param mean_abs_dev: (Boolean) To use the mean absolute deviation or traditional standard deviation.
        """
        self.prices = prices
        self.rets = np.log(prices).diff().dropna()
        self.window = window
        self.mean_abs_dev = mean_abs_dev

        # Properties relating to the tail sets
        self.vol_adj_rets = None

        # Compute tail sets
        self._vol_adjusted_rets()
        self.tail_sets = self.vol_adj_rets.dropna().apply(self._extract_tail_sets, axis=1)
        self.positive_tail_set = self.tail_sets.apply(self._positive_tail_set, axis=1)
        self.negative_tail_set = self.tail_sets.apply(self._negative_tail_set, axis=1)

    def get_tail_sets(self):
        """
        Computes the tail sets (positive and negative) and then returns a tuple with 3 elements, positive set, negative
        set, full matrix set.

        The positive and negative sets are each a series of lists with the names of the securites that fall within each
        set at a specific timestamp.

        For the full matirx a value of 1 indicates the volatility adjusted returns were in the upper decile, a value of
        -1 for the bottom decile.

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

        # Have 2 measure of vol, the mean absolute, and stdev
        if self.mean_abs_dev:
            # Huffman and Moll (2011) show that risk measured as the mean absolute deviation has more explanatory
            # power for future expected returns than standard deviation.
            vol = self.rets.abs().ewm(span=window, min_periods=window).mean()
        else:
            vol = self.rets.rolling(window).std()

        # Save vol adj rets
        self.vol_adj_rets = self.rets / vol

    @staticmethod
    def _extract_tail_sets(row):
        """
        Method used in a .apply() setting to transform each row in a DataFrame to the positive and negative tail sets.

        Currently this method splits the data using deciles (10 groups).

        :param row: (Series) vol adjusted returns for a given date.
        :return: (Series) Tail set with positive and negative labels.
        """
        # Get decile labels
        row_deciles = pd.qcut(x=row, q=10, labels=range(1, 11), retbins=False)

        # Set class labels
        row_deciles = row_deciles.to_numpy()  # Convert to numpy array
        row_deciles[(row_deciles != 1) & (row_deciles != 10)] = 0
        row_deciles[row_deciles == 1] = -1
        row_deciles[row_deciles == 10] = 1

        # Convert to series
        row_deciles = pd.Series(row_deciles, index=row.index)

        return row_deciles

    @staticmethod
    def _positive_tail_set(row):
        """
        Takes as input a row from the vol_adj_price DataFrame and then returns a list of names of the securites in the
        positive tail set, for this specific row date.

        This method is used in an apply() setting.

        :param row: (Series) of volatility adjusted prices.
        :return: (list) of securities in the positive tail set.
        """
        return list(row[row == 1].index)

    @staticmethod
    def _negative_tail_set(row):
        """
        Takes as input a row from the vol_adj_price DataFrame and then returns a list of names of the securites in the
        negative tail set, for this specific row date.

        This method is used in an apply() setting.

        :param row: (Series) of volatility adjusted prices.
        :return: (list) of securities in the negative tail set.
        """
        return list(row[row == -1].index)
