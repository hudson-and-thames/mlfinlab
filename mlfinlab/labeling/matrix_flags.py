"""
HEADER MENTION PAPER HERE
"""

import pandas as pd
import numpy as np


class MatrixFlagLabels:

    def __init__(self, data, window):
        # PUT ASSERTIONS THAT LEN OF DATA AND WINDOW MUST BE AT LEAST 10 #
        self.data = data
        self.window = window
        self.template = pd.DataFrame([[.5, 0, -1, -1, -1, -1, -1, -1, -1, 0],  # Bull flag template
                                      [1, 0.5, 0, -0.5, -1, -1, -1, -1, -0.5, 0],
                                      [1, 1, 0.5, 0, -0.5, -0.5, -0.5, -0.5, 0, 0.5],
                                      [0.5, 1, 1, 0.5, 0, -0.5, -0.5, -0.5, 0, 1],
                                      [0, 0.5, 1, 1, 0.5, 0, 0, 0, 0.5, 1],
                                      [0, 0, 0.5, 1, 1, 0.5, 0, 0, 1, 1],
                                      [-0.5, 0, 0, 0.5, 1, 1, 0.5, 0.5, 1, 1],
                                      [-0.5, -1, 0, 0, 0.5, 1, 1, 1, 1, 0],
                                      [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 0, -2],
                                      [-1, -1, -1, -1, -1, 0, 0.5, 0.5, -2, -2.5]])

    def set_template(self, template):
        """
        :param template: (pd.DataFrame) Template to override the default template. Must be a 10 by 10 pd.DataFrame.
        """
        self.template = template

    def transform_data(self, row_num, window=30):
        """
        :param row_num: (int) Row number to use for the "current" data point to apply the window to. The data window
                        contains the row corresponding to row_num, as well as the (self.window-1) preceding rows.
        :param window: (int) The number of rows preceding the current one to use for window. Override with
                        self.window in most cases.
        :return: (pd.DataFrame) The data window is split into 10 buckets each containing a chronological tenth of the
                        data window. Each point in 1 bucket is put into a decile corresponding to a position in a column based
                        on percentile relative to the entire data window. Bottom 10% on lowest row, next 10% on second
                        lowest row etc. THe proportion of points in each decile is reported to finalize the column.
                        The first tenth of the data is transformed to the leftmost column, the next
                        tenth to the next column on the right and so on until finally a 10 by 10 matrix is achieved.
        """
        # The relevant data to create matrix for current day consists of the day's return and the window number of days
        # preceding.
        data_window = self.data[row_num - window: row_num]

        # Find values for cutoff percentiles (deciles) over the entire data window. percentile_cutoffs is a list of
        # cutoffs for the 10th, 20th, 30th etc. percentiles in the data window.
        percentile_cutoffs = []
        for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            percentile_cutoffs.append(np.percentile(data_window, i))

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
            column = [np.count_nonzero(tenth_bins == 9), np.count_nonzero(tenth_bins == 8),
                      np.count_nonzero(tenth_bins == 7), np.count_nonzero(tenth_bins == 6),
                      np.count_nonzero(tenth_bins == 5), np.count_nonzero(tenth_bins == 4),
                      np.count_nonzero(tenth_bins == 3), np.count_nonzero(tenth_bins == 2),
                      np.count_nonzero(tenth_bins == 1), np.count_nonzero(tenth_bins == 0)]
            matrix[col_num] = column
            col_num += 1

        return matrix

    def apply_template_to_matrix(self, matrix, template):
        """
        :param matrix: (pd.DataFrame) Processed 10 by 10 matrix, where each column represents a chronological tenth
                        of the data, and each row represents a decile relative to the entire data window.
        :param template: (pd.DataFrame) Template to apply the processed matrix to.
        :return: (float) The total score for the day. Consists of the sum of sum of columns of the result from
                    multiplying the matrix element-wise with the template.
        """
        new_mat = matrix * template
        total_fit = sum(new_mat.sum(axis=0))
        return total_fit

    def apply_labeling_matrix(self):
        """
        :return: (pd.Series) Flag scores for the data series on each eligible day (meaning for indices self.window and
                    onwards).
        """
        labels = []
        idx = self.data[self.window:len(self.data)].index
        for row_num in range(self.window, len(self.data)):
            weights_matrix = self.transform_data(row_num=row_num, window=self.window)
            label = self.apply_template_to_matrix(weights_matrix, self.template)
            labels.append(label)

        return pd.Series(data=labels, index=idx)
