"""
Eigenportfolio applications
"""
import pandas as pd
import numpy as np
from .base import StatArb


class Eigenportfolio(StatArb):
    """
    Implements Eigenportfolio.

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
        self.eigenportfolio = None
        self.pc_num = None
        self.pca = None
        super(Eigenportfolio, self).__init__()

    def allocate(self, data, pc_num=0, window=0, intercept=True):
        """
        Calculate eigenportfolios for a given data, number of principal components, and window value.

        :param data: (pd.DataFrame) Time series of different assets.
        :param pc_num: (int) Number of principal components. (Default) 0, if using all components.
        :param window: (int) Length of rolling windows. (Default) 0 if no rolling.
        :param intercept: (bool) Indicates presence of intercept for the regression. (Default) True.
        """
        # Check some conditions.
        # self._check(data, pc, window, intercept)

        # Set pc, intercept and window.
        self.pc_num = pc_num
        self.intercept = intercept
        self.window = window

        # Set index, column, and price data.
        self.idx = data.index
        self.col = data.columns
        self.price = data

        # Convert given prices to np.array of log returns.
        self.log_returns = self._calc_log_returns(data)

        # No rolling windows.
        if not self.window:
            # Calculate the projection and eigenvector from the PCA of data.
            self.pca, self.eigenportfolio = self._calc_pca(self.log_returns, self.pc_num)

            # If intercept is True, add a constant of 1 on the right side of np_x.
            if self.intercept:
                self.pca = self._add_constant(self.pca)

            # Calculate the beta coefficients for linear regression.
            self.beta = self._linear_regression(self.pca, self.log_returns)

            # Calculate spread of residuals.
            self.resid = self.log_returns - self.pca.dot(self.beta)

            # Calculate the cumulative sum of residuals.
            self.cum_resid = self.resid.cumsum(axis=0)

            # Calculate z-score.
            self.z_score = self._calc_zscore(self.cum_resid)

        # Convert log returns to pd.DataFrame.
        self.log_returns = pd.DataFrame(self.log_returns, index=self.idx, columns=self.col)

        # Convert beta.
        self._convert_beta(self.beta)

        # Convert residuals.
        self._convert_resid(self.resid)

        # Convert cumulative residuals.
        self._convert_cum_resid(self.cum_resid)

        # Convert z_score.
        self._convert_zscore(self.z_score)

        # Convert eigenportfolio.
        self._convert_eigenportfolio(self.eigenportfolio)

    def _convert_eigenportfolio(self, eigenportfolio):
        """
        Converts given np.array of eigenportfolios to pd.DataFrame.

        :param eigenportfolio: (np.array) Eigenportfolio calculated by PCA.
        """
        # Index name for eigenportfolio.
        eigen_idx = []

        # Set index name.
        for i in range(self.pc_num):
            eigen_idx.append('eigenportfolio {}'.format(i))

        # Set self.eigenportfolio.
        self.eigenportfolio = pd.DataFrame(eigenportfolio, index=self.col, columns=eigen_idx).T

    def _convert_zscore(self, z_score):
        """
        Converts given np.array of z_score to pd.DataFrame.

        :param z_score: (np.array) Z-score calculated from cumulative residuals.
        """
        self.z_score = pd.DataFrame(z_score, index=self.idx, columns=self.col)

    def _convert_cum_resid(self, cum_resid):
        """
        Converts given np.array of cumulative residuals to pd.DataFrame.

        :param cum_resid: (np.array) Cumulative residuals from Regression.
        """
        self.cum_resid = pd.DataFrame(cum_resid, index=self.idx, columns=self.col)

    def _convert_resid(self, resid):
        """
        Converts given np.array of residuals to pd.DataFrame.

        :param resid: (np.array) Residuals from Regression.
        """
        self.resid = pd.DataFrame(resid, index=self.idx, columns=self.col)

    def _convert_beta(self, beta):
        """
        Converts given np.array of beta coefficients to pd.DataFrame.

        :param beta: (np.array) Beta coefficient of given regression.
        """
        # Index name for beta.
        beta_idx = []

        # Set index name.
        for i in range(self.pc_num):
            beta_idx.append('weight {}'.format(i))

        # Add constants row if there are intercepts.
        if self.intercept:
            beta_idx.append('constants')

        # Set self.beta.
        self.beta = pd.DataFrame(beta, index=beta_idx, columns=self.col)

    @staticmethod
    def _calc_pca(data, num):
        """
        Calculates the PCA projection of the data onto the n-top components.

        :param data: (np.array) Data to be projected.
        :param num: (int) Number of top-principal components.
        :return: (tuple) (np.array) Projected data, (np.array) Eigenvectors
        """
        # Standardize the data.
        data = (data - data.mean(axis=0)) / np.std(data, axis=0)

        # Calculate the covariance matrix.
        cov = data.T.dot(data) / data.shape[0]

        # Calculate the eigenvalue and eigenvector.
        eigval, eigvec = np.linalg.eigh(cov)

        # Get the index by sorting eigenvalue in descending order.
        idx = np.argsort(eigval)[::-1]

        # Sort eigenvector according to principal components.
        eigvec = eigvec[:, idx[:num]]

        # Projected data and eigenvector.
        return data.dot(eigvec), eigvec

    @staticmethod
    def _calc_log_returns(price):
        """
        Calculate the log returns for the given price data.

        :param price: (pd.DataFrame) Time series of prices.
        :return: (np.array) Log returns for the given price data.
        """
        # Convert to log.
        res = np.log(price)

        # Take the difference and replace the first value with 0.
        res = res.diff().fillna(0)

        # Conver to np.array and reshape to make a column format.
        res = np.array(res).reshape(res.shape)

        return res


def calc_rolling_eigenportfolio(data, num, window):
    """
    Calculate the rolling residuals and eigenportfolio for the number of principal components and
    number of rolling windows.

    :param data: (pd.DataFrame) User given data.
    :param num: (int) Number of top-principal components.
    :param window: (int) Number of rolling window.
    :return: (pd.DataFrame) The residuals and eigenportfolio of the given data and principal components.
    """
    # Change data into log returns.
    data = np.log(data).diff().fillna(0)

    # Convert to np.array.
    np_data = np.array(data)

    # Rolled data.
    # data = _rolling_window(np_data, window)

    return


def _calc_rolling_eig_params(data, num):
    """
    Helper function to calculate rolling eigenportfolio parameters.

    :param data: (np.array) Rolling window of original data.
    :return: (np.array) Data_x, data_y, beta, constant, spread, cum_resid, and z-score.
    """

    # Calculate beta, the slope and intercept.
    try:
        beta = np.linalg.inv(np_x.T.dot(np_x)).dot(np_x.T).dot(np_y)
    except:
        beta = np.linalg.pinv(np_x.T.dot(np_x)).dot(np_x.T).dot(np_y)

    # Calculate spread.
    spread = np_y - np_x.dot(beta)

    # Calculate cumulative sum of spread of returns.
    cum_resid = spread.cumsum()

    # Calculate z-score.
    z_score = (cum_resid[-1] - np.mean(cum_resid)) / np.std(cum_resid)

    # Separate the resulting array.
    res = np.array([beta[0][0], beta[1][0], spread[-1][0], cum_resid[-1], z_score])

    return res
