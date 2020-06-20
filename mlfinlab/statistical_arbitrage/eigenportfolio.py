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
        self.pc = None
        self.window = None
        self.intercept = None

    def allocate(self, data, pc=0, window=0, intercept=True):
        """
        Calculate eigenportfolios for a given data, number of principal components, and window value.

        :param data: (pd.DataFrame) Time series of different assets.
        :param pc: (int) Number of principal components. (Default) 0, if using all components.
        :param window: (int) Length of rolling windows. (Default) 0 if no rolling.
        :param intercept: (bool) Indicates presence of intercept for the regression. (Default) True.
        """
        # Check some conditions.
        # self._check(data, pc, window, intercept)

        # Set pc, intercept and window.
        self.pc = pc
        self.intercept = intercept
        self.window = window

        # Convert given prices to np.array of log returns.
        np_data = self._calc_log_returns(data)

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
        res = np.array(np.log(res).diff().fillna(0)).reshape(res.shape)

        return res



    # # Change data into log returns.
    # data = np.log(data).diff().fillna(0)
    #
    # # Calculate the projection and eigenvector from the PCA of data.
    # data_proj, data_eigvec = calc_pca(data, num)
    #
    # # Add a constant of 1 on the right side for data_proj to account for intercepts.
    # data_proj = np.hstack((data_proj, np.ones((data_proj.shape[0], 1))))
    #
    # # Linear regression by matrix multiplication.
    # beta = np.linalg.inv(data_proj.T.dot(data_proj)).dot(data_proj.T).dot(np.array(data))
    #
    # # Calculate spread.
    # spread = data - data_proj.dot(beta)
    #
    # # Calculate cumulative sum of spread of returns.
    # cum_resid = spread.cumsum()
    #
    # # Calculate z-score.
    # z_score = (cum_resid - np.mean(cum_resid)) / np.std(cum_resid)
    #
    # # Index name for beta.
    # beta_idx = []
    #
    # # Index name for eigenportfolio.
    # eigenp_idx = []
    #
    # # Set index name.
    # for i in range(beta.shape[0] - 1):
    #     beta_idx.append('weight {}'.format(i))
    #     eigenp_idx.append('eigenportfolio {}'.format(i))
    # beta_idx.append('constants')
    #
    # # Conver to pd.DataFrame.
    # beta = pd.DataFrame(beta, index=beta_idx, columns=data.columns)
    # data_eigvec = pd.DataFrame(data_eigvec.T, index=eigenp_idx, columns=data.columns)
    #
    # # Combine all dataframes.
    # combined_df = pd.concat([data, data_eigvec, beta, spread, cum_resid, z_score], axis=0,
    #                         keys=['log_ret', 'eigenportfolio', 'beta', 'ret_spread', 'cum_resid',
    #                               'z_score'])
    # return combined_df


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
    #data = _rolling_window(np_data, window)


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

