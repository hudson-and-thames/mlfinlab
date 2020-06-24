# pylint: disable=no-self-use
# pylint: disable=unnecessary-comprehension
"""
Matrix Flag labeling method.
"""

import pandas as pd
import numpy as np


class MatrixFlagLabels:
    """
    The Matrix Flag labeling method is featured in the paper: Cervelló-Royo, R., Guijarro, F. and Michniuk, K., 2015.
    Stock market trading rule based on pattern recognition and technical analysis: Forecasting the DJIA index with
    intraday data.

    The method of applying a matrix template was first introduced, and explained in greater detail, in the paper:
    Leigh, W., Modani, N., Purvis, R. and Roberts, T., 2002. Stock market trading rule discovery using technical
    charting heuristics.

    Cervelló-Royo et al. expand on Leigh et al.'s work by proposing a new bull flag pattern which ameliorates some
    weaknesses in Leigh's original template. Additionally, he applies this bull flag labeling method to intraday
    candlestick data, rather than just closing prices.

    To find the total weight for a given day, the current price as well as the preceding window days number of prices is
    used. The data window is split into 10 buckets each containing a chronological tenth of the data window. Each point
    in 1 bucket is put into a decile corresponding to a position in a column based on percentile relative to the entire
    data window. Bottom 10% on lowest row, next 10% on second lowest row etc.
    The proportion of points in each decile is reported to finalize the column. The first tenth of the data is
    transformed to the leftmost column, the next tenth to the next column on the right and so on until finally a 10 by
    10 matrix is achieved. This matrix is then multiplied element-wise with the 10 by 10 template, and the sum of all
    columns is the total weight for the day. If desired, the user can specify a threshold to determine positive and
    negative classes. The value of the threshold depends on how strict of a classifier the user desires, and the
    allowable values based on the template matrix.
    """

    def __init__(self, prices, window, template_name=None):
        """
        :param prices: (pd.Series) Price data for one stock.
        :param window: (int) Length of preceding data window used when generating the fit matrix for one day.
        :param template_name: (str) Name of the an available template in the template library. Allowable names:
                            ``leigh_bear``, ``leigh_bull``, ``cervelloroyo_bear``, ``cervellororo_bull``.
        """
        assert (len(prices) >= 10), "Length of data must be at least 10."
        assert (window >= 10), "Window must be at least 10."
        assert (len(prices) >= window), "Window cannot be greater than length of data."
        assert isinstance(prices, pd.Series), "Data must be pd.Series."
        self.data = prices
        self.window = window
        self.template = pd.DataFrame()

        if template_name is not None:
            self._init_template(template_name)

    def _init_template(self, name):
        """
        :param name: (str) Name of the an available template in the template library. Allowable names: ``leigh_bear``,
        ``leigh_bull``, ``cervelloroyo_bear``, ``cervellororo_bull``.
        """
        leigh_bull = pd.DataFrame([[.5, 0, -1, -1, -1, -1, -1, -1, -1, 0],
                                   [1, 0.5, 0, -0.5, -1, -1, -1, -1, -0.5, 0],
                                   [1, 1, 0.5, 0, -0.5, -0.5, -0.5, -0.5, 0, 0.5],
                                   [0.5, 1, 1, 0.5, 0, -0.5, -0.5, -0.5, 0, 1],
                                   [0, 0.5, 1, 1, 0.5, 0, 0, 0, 0.5, 1],
                                   [0, 0, 0.5, 1, 1, 0.5, 0, 0, 1, 1],
                                   [-0.5, 0, 0, 0.5, 1, 1, 0.5, 0.5, 1, 1],
                                   [-0.5, -1, 0, 0, 0.5, 1, 1, 1, 1, 0],
                                   [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 0, -2],
                                   [-1, -1, -1, -1, -1, 0, 0.5, 0.5, -2, -2.5]])

        leigh_bear = pd.DataFrame(np.flip(np.array(leigh_bull), axis=0))
        cervelloroyo_bull = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                                          [0, 0, 0, -1, -2, -2, -2, -2, -2, -2],
                                          [0, 0, -1, -3, -3, -3, -3, -3, -3, -3],
                                          [0, -1, -3, -5, -5, -5, -5, -5, -5, -5],
                                          [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
                                          [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
                                          [5, -1, -5, -5, -5, -5, -5, -5, -5, -5]])
        cervelloroyo_bear = pd.DataFrame(np.flip(np.array(cervelloroyo_bull), axis=0))

        if name == 'leigh_bull':
            self.set_template(leigh_bull)
        elif name == 'leigh_bear':
            self.set_template(leigh_bear)
        elif name == 'cervelloroyo_bear':
            self.set_template(cervelloroyo_bear)
        elif name == 'cervelloroyo_bull':
            self.set_template(cervelloroyo_bull)
        else:
            raise Exception("Invalid template name. Valid names are 'leigh_bull', 'leigh_bear', "
                            "'cervelloroyo_bear', 'cervelloroyo_bull'.")

    def set_template(self, template):
        """
        :param template: (pd.DataFrame) Template to override the default template. Must be a 10 by 10 pd.DataFrame.
                            NaN values not allowed, as they will not automatically be treated as zeros.
        """
        assert template.shape == (10, 10), "Template must be 10 by 10."
        assert not template.isnull().values.any(), "No NaN values allowed in template."
        self.template = template

    def _transform_data(self, row_num, window=30):
        """
        :param row_num: (int) Row number to use for the "current" data point to apply the window to. The data window
                        contains the row corresponding to row_num, as well as the (self.window-1) preceding rows.
        :param window: (int) The number of rows preceding the current one to use for window. Override with
                        self.window in most cases.
        :return: (pd.DataFrame) Transformed 10 by 10 matrix, in which each column corresponds to a chronological tenth
                    of the data window, and each row corresponds to a price decile relative to the entire data window.
                    The template matrix is then applied to this output matrix.
        """
        # The relevant data to create matrix for current day consists of the day's return and the window number of days
        # preceding.
        data_window = self.data[row_num - window: row_num]

        # Find values for cutoff percentiles (deciles) over the entire data window. percentile_cutoffs is a list of
        # cutoffs for the 10th, 20th, 30th etc. percentiles in the data window.
        percentiles = np.linspace(10, 100, num=10)
        percentile_cutoffs = np.percentile(data_window, percentiles)

        # Each value in bins corresponds to the value of the same index in data_window. If bin is 0, the corresponding
        # point in data_window is in the 0-10th percentile, if 1 then 10-20th percentile, ..., if 9 then 90-100th
        # percentile.
        bins = np.digitize(data_window, bins=percentile_cutoffs, right=True)

        # Dictionary to map values in data window to their percentile bin.
        value_to_bin = dict(zip(data_window, bins))

        # Split data window into 10 chronological sub-windows.
        data_split = np.array_split(data_window, 10)

        # Create the matrix, one column at a time from left to right. The leftmost column corresponds to the first
        # tenth of data, and the rightmost corresponds to the final tenths. The top element of each column corresponds
        # to the count in the tenth in the 90-100th percentile of data window, second from top is count in the tenth
        # in the 80-90th percentile and so on until the last element is the count in the 0-10th percentile.
        matrix = pd.DataFrame()
        col_num = 0  # Left most column (out of 10) to populate using the first tenth of data.
        for tenth in data_split:
            # Apply the dictionary to data to get the bins each data point belongs to.
            tenth_bins = np.vectorize(value_to_bin.get)(tenth)
            # We count the number of times each bin appears in each tenth of data, starting from the highest decile.
            column, _ = np.histogram(tenth_bins, bins=[i for i in range(11)])
            # Convert the count in the column to proportion.
            column = np.array(column) / len(tenth)
            matrix[col_num] = column[::-1]  # Reverse so that highest decile is on top of the column.
            col_num += 1

        return matrix

    def _apply_template_to_matrix(self, matrix, template):
        """
        :param matrix: (pd.DataFrame) Processed 10 by 10 matrix, where each column represents a chronological tenth
                        of the data, and each row represents a decile relative to the entire data window.
        :param template: (pd.DataFrame) Template to apply the processed matrix to.
        :return: (float) The total score for the day. Consists of the sum of sum of columns of the result from
                    multiplying the matrix element-wise with the template.
        """
        new_mat = matrix * template
        total_fit = new_mat.values.sum()
        return total_fit

    def apply_labeling_matrix(self, threshold=None):
        """
        :param threshold: (float) If None, labels will be returned numerically as the score for the day. If not None,
                        then labels are returned categorically, with the positive category for labels that are equal to
                        or exceed the threshold.
        :return: (pd.Series) Total scores for the data series on each eligible day (meaning for indices self.window and
                    onwards).
        """
        labels = []
        idx = self.data[self.window:len(self.data)].index

        # Apply the data transformation and template.
        for row_num in range(self.window, len(self.data)):
            weights_matrix = self._transform_data(row_num=row_num, window=self.window)
            label = self._apply_template_to_matrix(weights_matrix, self.template)
            labels.append(label)

        # If threshold is given, replace fit values by whether they exceed the threshold.
        if threshold is not None:
            labels = [1 if i >= threshold else -1 for i in labels]
        return pd.Series(data=labels, index=idx)
