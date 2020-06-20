# pylint: disable=bare-except
"""
Calculate pairs regression.
"""

import numpy as np
import pandas as pd


class StatArb:
    """
    Implements Statistical Arbitrage.

    Possible outputs are:

    - ``self.price`` (pd.DataFrame) Price of the given assets.
    - ``self.log_returns`` (pd.DataFrame) Log returns calculated by given price data.
    - ``self.beta`` (pd.DataFrame) Regression parameters.
    - ``self.resid`` (pd.DataFrame) Residuals from Regression.
    - ``self.cum_resid`` (pd.DataFrame) Cumulative residuals from Regression.
    - ``self.z_score`` (pd.DataFrame) Z-score calculated from cumulative residuals.
    - ``self.intercept`` (bool) Checks to include constant term for regression. (Default) True.
    - ``self.window`` (int) Number of window to roll. (Default) 0 if no rolling.
    """

    def __init__(self):
        self.price = None
        self.log_returns = None
        self.beta = None
        self.resid = None
        self.cum_resid = None
        self.z_score = None
        self.intercept = False
        self.window = 0
        self.idx = None
        self.col = None
        # self.signal = None (To be implemented)

    def allocate(self, price_x, price_y, window=0, intercept=True):
        """
        Calculate pairs trading for price_x on price_y.

        :param price_x: (pd.Series) Time series of price of x (DO NOT adjust for log).
        :param price_y: (pd.Series) Time series of price of y (DO NOT adjust for log).
        :param window: (int) Number of window to roll. (Default) 0 if no rolling.
        :param intercept: (bool) Indicates presence of intercept for the regression. (Default) True.
        """
        # Check some conditions.
        # self._check(price_x, price_y, window, intercept)

        # Set intercept and window.
        self.intercept = intercept
        self.window = window

        # Convert given prices to np.array of log returns.
        np_x = self._calc_log_returns(price_x)
        np_y = self._calc_log_returns(price_y)

        # If intercept is True, add a constant of 1 on the right side of np_x.
        if self.intercept:
            np_x = self._add_constant(np_x)

        # No rolling windows.
        if not self.window:
            # Calculate the beta coefficients for linear regression.
            self.beta = self._linear_regression(np_x, np_y)

            # Calculate spread of residuals.
            self.resid = np_y - np_x.dot(self.beta)

            # Calculate the cumulative sum of residuals.
            self.cum_resid = self.resid.cumsum(axis=0)

            # Calculate z-score.
            self.z_score = self._calc_zscore(self.cum_resid)
        else:
            self._rolling_allocate(np_x, np_y)

        # Convert prices to pd.DataFrame.
        self._convert_price(price_x, price_y)

        # Convert log returns to pd.DataFrame.
        self._convert_log_returns(np_x, np_y)

        # Convert beta.
        self._convert_beta(self.beta)

        # Convert residuals.
        self._convert_resid(self.resid)

        # Convert cumulative residuals.
        self._convert_cum_resid(self.cum_resid)

        # Convert z_score.
        self._convert_zscore(self.z_score)

    def _rolling_allocate(self, np_x, np_y):
        """
        Calculate allocate with a rolling window.

        :param np_x: (np.array) Log returns of price_x.
        :param np_y: (np.array) Log returns of price_y.
        """
        # Preset variables.
        self.beta = np.zeros((self.intercept+1, np_x.shape[0]))
        self.resid = np.zeros((np_x.shape[0], 1))
        self.cum_resid = np.zeros((np_x.shape[0], 1))
        self.z_score = np.zeros((np_x.shape[0], 1))

        # Combine np_x and np_y.
        np_xy = np.hstack((np_x, np_y))

        # Rolling combined data.
        np_xy = self._rolling_window(np_xy, self.window)

        # Fill in the array.
        for it in range(np_xy.shape[0]):
            self._calc_rolling_params(np_xy[it], it + self.window - 1)

        # Set np.nan for values before the initial window.
        self.beta[:, :self.window - 1] = np.nan
        self.resid[:self.window - 1] = np.nan
        self.cum_resid[:self.window - 1] = np.nan
        self.z_score[:self.window - 1] = np.nan

    def _calc_rolling_params(self, np_xy, adj_window):
        """
        Helper function to calculate rolling regression parameters.

        :param np_xy: (np.array) Rolling window of combined x and y log returns.
        :param adj_window: (int) Adjusted window to set values for the arrays.
        """
        # Split data to np_x.
        np_x = np_xy[:, :-1]

        # Split data to np_y.
        np_y = np_xy[:, [-1]]

        # Calculate the beta coefficients for linear regression.
        beta = self._linear_regression(np_x, np_y)

        # Calculate spread of residuals.
        resid = np_y - np_x.dot(beta)

        # Calculate the cumulative sum of residuals.
        cum_resid = resid.cumsum(axis=0)

        # Calculate and set z-score.
        self.z_score[adj_window] = self._calc_zscore(cum_resid)[-1]

        # Insert beta.
        self.beta[:, [adj_window]] = beta

        # Insert resid.
        self.resid[adj_window] = resid[-1]

        # Insert cum_resid.
        self.cum_resid[adj_window] = cum_resid[-1]

    @staticmethod
    def _rolling_window(data, window):
        """
        Helper function to generate rolling windows.

        :param data: (np.array) Original data given by user.
        :param window: (int) Number of rolling window.
        :return: (np.array) All generated windows.
        """
        shape = (data.shape[0] - window + 1, window) + data.shape[1:]
        strides = (data.strides[0],) + data.strides
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def _convert_zscore(self, z_score):
        """
        Converts given np.array of z_score to pd.DataFrame.

        :param z_score: (np.array) Z-score calculated from cumulative residuals.
        """
        self.z_score = pd.DataFrame(z_score, index=self.idx, columns=['Z-Score'])

    def _convert_cum_resid(self, cum_resid):
        """
        Converts given np.array of cumulative residuals to pd.DataFrame.

        :param cum_resid: (np.array) Cumulative residuals from Regression.
        """
        self.cum_resid = pd.DataFrame(cum_resid, index=self.idx, columns=['Cumulative Residuals'])

    def _convert_resid(self, resid):
        """
        Converts given np.array of residuals to pd.DataFrame.

        :param resid: (np.array) Residuals from Regression.
        """
        self.resid = pd.DataFrame(resid, index=self.idx, columns=['Residuals'])

    def _convert_beta(self, beta):
        """
        Converts given np.array of beta coefficients to pd.DataFrame.

        :param beta: (np.array) Beta coefficient of given regression.
        """
        if self.intercept:
            self.beta = pd.DataFrame(beta, index=['beta', 'constant']).T
        else:
            self.beta = pd.DataFrame(beta, index=['beta']).T

    def _convert_log_returns(self, np_x, np_y):
        """
        Converts given np.array of log returns data to combined pd.DataFrame.

        :param np_x: (np.array) Log returns of price_x.
        :param np_y: (np.array) Log returns of price_y.
        """
        self.log_returns = pd.DataFrame([np_x[:, 0], np_y[:, 0]], index=self.col,
                                        columns=self.idx).T

    def _convert_price(self, price_x, price_y):
        """
        Converts given x and y price data to combined pd.DataFrame.

        :param price_x: (pd.Series) Time series of price of x.
        :param price_y: (pd.Series) Time series of price of y.
        """
        self.price = pd.DataFrame([price_x, price_y]).T

        # Set index and column from self.price.
        self.idx = self.price.index
        self.col = self.price.columns

    @staticmethod
    def _calc_zscore(data):
        """
        Calculates the z-score for the given data.

        :param data: (np.array) Data for z-score calculation.
        :return: (np.array) Z-score of the given data.
        """
        return (data - np.mean(data)) / np.std(data)

    @staticmethod
    def _linear_regression(data_x, data_y):
        """
        Calculates the parameter vector using matrix multiplication.

        :param data_x: (np.array) Time series of log returns of x.
        :param data_y: (np.array) Time series of log returns of y.
        :return: (np.array) Parameter vector.
        """
        try:
            beta = np.linalg.inv(data_x.T.dot(data_x)).dot(data_x.T).dot(data_y)
        except:
            beta = np.linalg.pinv(data_x.T.dot(data_x)).dot(data_x.T).dot(data_y)
        return beta

    @staticmethod
    def _add_constant(returns):
        """
        Adds a constant of 1 on the right side of the given returns.

        :param returns: (np.array) Log returns for a given time series.
        :return: (np.array) Log returns with an appended column of 1 on the right.
        """
        #  Adds a column of 1 on the right side of the given array.
        return np.hstack((returns, np.ones((returns.shape[0], 1))))

    @staticmethod
    def _calc_log_returns(price):
        """
        Calculate the log returns for the given price data.

        :param price: (pd.Series) Time series of prices.
        :return: (np.array) Log returns for the given price data.
        """
        # Convert to log.
        res = np.log(price)

        # Take the difference and replace the first value with 0.
        res = res.diff().fillna(0)

        # Conver to np.array and reshape to make a column format.
        res = np.array(res).reshape((price.size, 1))

        return res
