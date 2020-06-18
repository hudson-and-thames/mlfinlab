"""
Principal Component Analysis applied for Statistical Arbitrage.
"""
import pandas as pd
import numpy as np

from .regression import _calc_rolling_params, _rolling_window


def calc_all_eigenportfolio(data, num):
    """
    Calculate the residuals and eigenportfolio for the number of principal components.

    :param data: (pd.DataFrame) User given data.
    :param num: (int) Number of top-principal components.
    :return: (pd.DataFrame) The residuals and eigenportfolio of the given data and principal components.
    """
    # Calculate the projection and eigenvector from the PCA of data.
    data_proj, data_eigvec = calc_pca(data, num)

    # Add a constant of 1 on the right side for data_proj to account for intercepts.
    data_proj = np.hstack((data_proj, np.ones((data_proj.shape[0], 1))))

    # Linear regression by matrix multiplication.
    beta = np.linalg.inv(data_proj.T.dot(data_proj)).dot(data_proj.T).dot(np.array(data))

    # Calculate residuals.
    resid = data - data_proj.dot(beta)

    # Convert residuals to pd.DataFrame.
    resid = pd.DataFrame(resid, index=data.index, columns=data.columns)

    # Convert beta, the eigenportfolio and constant ratio, to pd.DataFrame.
    idx = []
    for i in range(beta.shape[0] - 1):
        idx.append('Eigenportfolio {}'.format(i))
    idx.append('Constants')
    beta = pd.DataFrame(beta, index=idx, columns=data.columns)

    combined_df = pd.concat([resid, beta], axis=0, keys=['Spread', 'Eigenportfolio'])
    return combined_df


def calc_rolling_eigenportfolio(data, num, window):
    """
    Calculate the rolling residuals and eigenportfolio for the number of principal components and
    number of rolling windows.

    :param data: (pd.DataFrame) User given data.
    :param num: (int) Number of top-principal components.
    :param window: (int) Number of rolling window.
    :return: (pd.DataFrame) The residuals and eigenportfolio of the given data and principal components.
    """
    # Convert to np.array.
    np_data = np.array(data)

    # Rolled data.
    np_data = _rolling_window(np_data, window)

    return


def calc_pca(data, num):
    """
    Calculates the PCA projection of the data onto the num-top components.

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
    return eigvec.T.dot(data.T).T, eigvec
