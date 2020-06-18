"""
Principal Component Analysis applied for Statistical Arbitrage.
"""
import pandas as pd
import numpy as np


def calc_all_pca():
    return


def calc_pca(data, num):
    """
    Calculates the PCA project of the data onto the n-top components.

    :param data: (np.array) Data to be projected.
    :param num: (int) Number of top-principal components.
    :return: (tuple) (np.array) Projected data, (np.array) Eigenvectors
    """
    # Standardize the data.
    data = data - data.mean(axis=0) / np.std(data, axis=0)

    # Calculate the covariance matrix.
    cov = data.T.dot(data) / data.shape[0]

    # Calculate the eigenvalue and eigenvector.
    eigval, eigvec = np.linalg.eigh(cov.T.dot(cov))

    # Get the index by sorting eigenvalue in descending order.
    idx = np.argsort(eigval)[::-1]

    # Sort eigenvector according to principal components.
    eigvec = eigvec[:, idx[:num]]

    # Projected data and eigenvector
    return eigvec.T.dot(data.T).T, eigvec
