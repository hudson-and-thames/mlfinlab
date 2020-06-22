"""
Implements Eigenportfolio.
"""
import pandas as pd
import numpy as np
import warnings
from .base import StatArb


class Eigenportfolio(StatArb):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=arguments-differ
    """
    .. warning::

        - The structure is designed to use the original price data for calculation. The class itself
          calculates the log returns for the arbitrage. Do NOT adjust your input for the logarithmic
          scale.

        - Use pd.DataFrame as inputs.

    Possible outputs are:

    - ``self.price`` (pd.DataFrame) Price of the given assets.
    - ``self.log_returns`` (pd.DataFrame) Log returns calculated by given price data.
    - ``self.beta`` (pd.DataFrame) Regression parameters.
    - ``self.resid`` (pd.DataFrame) Residuals from Regression.
    - ``self.cum_resid`` (pd.DataFrame) Cumulative residuals from Regression.
    - ``self.z_score`` (pd.DataFrame) Z-score calculated from cumulative residuals.
    - ``self.intercept`` (bool) Checks to include constant term for regression. (Default) True.
    - ``self.window`` (int) Number of window to roll. (Default) 0 if no rolling.
    - ``self.eigenportfolio`` (pd.DataFrame) Eigenportfolio calculated from PCA.
    - ``self.pc_num`` (int) Number of principal components.
    - ``self.pca`` (np.array) Projected log_returns data according to PCA.
    """

    def __init__(self):
        self.eigenportfolio = None
        self.pc_num = None
        self.pca = None
        super(Eigenportfolio, self).__init__()

    def allocate(self, data, pc_num, window=0, intercept=True):
        """
        Calculate eigenportfolios for a given data, number of principal components, and window value.

        :param data: (pd.DataFrame) Time series of different assets.
        :param pc_num: (int) Number of principal components.
        :param window: (int) Length of rolling windows. (Default) 0 if no rolling.
        :param intercept: (bool) Indicates presence of intercept for the regression. (Default) True.
        """
        # Check conditions.
        self._check(data, pc_num, window, intercept)

        # Set pc_num, intercept and window.
        self.pc_num = pc_num
        self.intercept = intercept
        self.window = window

        # Set index, column, price data, and number of assets.
        self.idx = data.index
        self.col = data.columns
        self.price = data
        self.num_assets = len(self.col)

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
        else:
            # Allocate with rolling windows.
            self._rolling_allocate(self.log_returns)

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

    @staticmethod
    def _check(*args):
        """
        Checks if the user given variables are correct.

        :param args: User given data, pc_num, window, intercept.
        """
        # Set values.
        data, pc_num, window, intercept = args[0], args[1], args[2], args[3]

        # Check if data is a pd.DataFrame.
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pd.DataFrame.")

        # Check that the given data has no null value.
        if data.isnull().any().sum() != 0:
            raise ValueError("Given data contains values of null. Remove the null values.")

        # Check that the given data has no values of 0.
        if (data == 0).any().sum() != 0:
            raise ValueError("Given data contains values of 0. Remove the 0 values.")

        # Check if pc_num is an integer.
        if not isinstance(pc_num, int):
            raise ValueError("Number of principal components must be an integer.")

        # Check if the range of pc_num is correct.
        if pc_num < 1 or pc_num > data.shape[1]:
            raise ValueError("Number of principal components must be between 1 and the number of "
                             "assets.")

        # Check if window is an integer.
        if not isinstance(window, int):
            raise ValueError("Length of window must be an integer.")

        # Check if the range of window is correct.
        if window < 0 or window > data.shape[0]:
            raise ValueError("Length of window must be between 0 and the number of "
                             "periods. 0 indicates using the entire data.")

        # Check if intercept is a boolean.
        if not isinstance(intercept, bool):
            raise ValueError("Intercept must be either True or False.")

    def _calc_rolling_params(self, data, adj_window):
        """
        Helper function to calculate rolling eigenportfolio parameters.

        :param data: (np.array) Rolling window of combined x and y log returns.
        :param adj_window: (int) Adjusted window to set values for the arrays.
        """
        pca, eigenportfolio = self._calc_pca(data, self.pc_num)

        # If intercept is True, add a constant of 1 on the right side of np_x.
        if self.intercept:
            pca = self._add_constant(pca)

        # Calculate the beta coefficients for linear regression.
        beta = self._linear_regression(pca, data)

        # Calculate spread of residuals.
        resid = data - pca.dot(beta)

        # Calculate the cumulative sum of residuals.
        cum_resid = resid.cumsum(axis=0)

        # Calculate and set z-score.
        self.z_score[adj_window] = self._calc_zscore(cum_resid)[-1]

        # Insert beta.
        self.beta[adj_window] = beta

        # Insert resid.
        self.resid[adj_window] = resid[-1]

        # Insert cum_resid.
        self.cum_resid[adj_window] = cum_resid[-1]

        # Insert eigenportfolio.
        self.eigenportfolio[adj_window] = eigenportfolio.T

    def _rolling_allocate(self, log_returns):
        """
        Calculate allocate with a rolling window.

        :param log_returns: (np.array) Log returns of the given data.
        """
        # Preset variables.
        self.beta = np.zeros((log_returns.shape[0], self.intercept + self.pc_num, self.num_assets))
        self.resid = np.zeros((log_returns.shape[0], self.num_assets))
        self.cum_resid = np.zeros((log_returns.shape[0], self.num_assets))
        self.z_score = np.zeros((log_returns.shape[0], self.num_assets))
        self.eigenportfolio = np.zeros((log_returns.shape[0], self.pc_num, self.num_assets))

        # Rolling combined data.
        log_returns = self._rolling_window(log_returns, self.window)

        # Fill in the array.
        for itr in range(log_returns.shape[0]):
            self._calc_rolling_params(log_returns[itr], itr + self.window - 1)

        # Set np.nan for values before the initial window.
        self.beta[:self.window - 1] = np.nan
        self.resid[:self.window - 1] = np.nan
        self.cum_resid[:self.window - 1] = np.nan
        self.z_score[:self.window - 1] = np.nan
        self.eigenportfolio[:self.window - 1] = np.nan

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
        if self.window:
            self.eigenportfolio = pd.concat([pd.DataFrame(b, index=eigen_idx, columns=self.col)
                                             for b in eigenportfolio], keys=self.idx)
        else:
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
        if self.window:
            self.beta = pd.concat([pd.DataFrame(b, index=beta_idx, columns=self.col)
                                   for b in self.beta], keys=self.idx)
        else:
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.nan_to_num((data - np.mean(data, axis=0)) / np.std(data, axis=0))

        # Calculate the covariance matrix.
        cov = data.T.dot(data) / data.shape[0]

        # Calculate the eigenvalue and eigenvector.
        eigval, eigvec = np.linalg.eigh(cov)

        # Get the index by sorting eigenvalue in descending order.
        idx = np.argsort(eigval)[::-1]

        # Sort eigenvector according to principal components.
        eigvec = eigvec[:, idx[:num]]

        # Scale eigenvector to leverage of 1.
        eigvec = eigvec / np.sum(np.abs(eigvec), axis=0)

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
