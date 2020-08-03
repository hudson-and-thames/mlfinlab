"""
An implementation of the Exact Fit of the first 3 Moments (EF3M) of finding the parameters that make up the mixture
of 2 Gaussian distributions. Based on the work by Lopez de Prado and Foreman (2014) "A mixture of two Gaussians
approach to mathematical portfolio oversight: The EF3M algorithm." Quantitative Finance, Vol. 14, No. 5, pp. 913-930.
"""

import sys
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import gaussian_kde
from numba import njit, objmode


class M2N:
    """
    M2N - A Mixture of 2 Normal distributions
    This class is used to contain parameters and equations for the EF3M algorithm, when fitting parameters to a mixture
    of 2 Gaussian distributions.

    :param moments: (list) The first five (1... 5) raw moments of the mixture distribution.
    :param epsilon: (float) Fitting tolerance
    :param factor: (float) Lambda factor from equations
    :param n_runs: (int) Number of times to execute 'singleLoop'
    :param variant: (int) The EF3M variant to execute, options are 1: EF3M using first 4 moments, 2: EF3M using
     first 5 moments
    :param max_iter: (int) Maximum number of iterations to perform in the 'fit' method
    :param num_workers: (int) Number of CPU cores to use for multiprocessing execution. Default is -1 which sets
     num_workers to all cores.

    """
    def __init__(self, moments, epsilon=10**-5, factor=5, n_runs=1, variant=1, max_iter=100_000, num_workers=-1):
        """
        Constructor

        :param moments: (list) The first five (1... 5) raw moments of the mixture distribution.
        :param epsilon: (float) Fitting tolerance
        :param factor: (float) Lambda factor from equations
        :param n_runs: (int) Number of times to execute 'singleLoop'
        :param variant: (int) The EF3M variant to execute, options are 1: EF3M using first 4 moments, 2: EF3M using
         first 5 moments
        :param max_iter: (int) Maximum number of iterations to perform in the 'fit' method
        :param num_workers: (int) Number of CPU cores to use for multiprocessing execution. Default is -1 which sets
         num_workers to all cores.

        The parameters of the mixture are defined by a list, where:
            parameters = [mu_1, mu_2, sigma_1, sigma_2, p_1]
        """

        pass

    def fit(self, mu_2):
        """
        Fits and the parameters that describe the mixture of the 2 Normal distributions for a given set of initial
        parameter guesses.

        :param mu_2: (float) An initial estimate for the mean of the second distribution.
        """

        pass

    def get_moments(self, parameters, return_result=False):
        """
        Calculates and returns the first five (1...5) raw moments corresponding to the newly estimated parameters.

        :param parameters: (list) List of parameters if the specific order [mu_1, mu_2, sigma_1, sigma_2, p_1]
        :param return_result: (bool) If True, method returns a result instead of setting the 'self.new_moments'
         attribute.
        :return: (list) List of the first five moments
        """

        pass

    def iter_4(self, mu_2, p_1):
        """
        Evaluation of the set of equations that make up variant #1 of the EF3M algorithm (fitting using the first
        four moments).

        :param mu_2: (float) Initial parameter value for mu_2
        :param p_1: (float) Probability defining the mixture; p_1, 1 - p_1
        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,
         divide-by-zero), otherwise an empty list is returned.
        """

        pass

    def iter_5(self, mu_2, p_1):
        """
        Evaluation of the set of equations that make up variant #2 of the EF3M algorithm (fitting using the first five
        moments).

        :param mu_2: (float) Initial parameter value for mu_2
        :param p_1: (float) Probability defining the mixture; p_1, 1-p_1
        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,
         divide-by-zero), otherwise an empty list is returned.
        """


        pass

    def single_fit_loop(self, epsilon=0):
        """
        A single scan through the list of mu_2 values, cataloging the successful fittings in a DataFrame.

        :param epsilon: (float) Fitting tolerance.
        :return: (pd.DataFrame) Fitted parameters and error
        """

        pass

    def mp_fit(self):
        """
        Parallelized implementation of the 'single_fit_loop' method. Makes use of dask.delayed to execute multiple
        calls of 'single_fit_loop' in parallel.

        :return: (pd.DataFrame) Fitted parameters and error
        """

        pass


# === Helper functions, outside the M2N class. === #
def centered_moment(moments, order):
    """
    Compute a single moment of a specific order about the mean (centered) given moments about the origin (raw).

    :param moments: (list) First 'order' raw moments
    :param order: (int) The order of the moment to calculate
    :return: (float) The central moment of specified order.
    """

    pass


def raw_moment(central_moments, dist_mean):
    """
    Calculates a list of raw moments given a list of central moments.

    :param central_moments: (list) The first n (1...n) central moments as a list.
    :param dist_mean: (float) The mean of the distribution.
    :return: (list) The first n+1 (0...n) raw moments.
    """

    pass


def most_likely_parameters(data, ignore_columns='error', res=10_000):
    """
    Determines the most likely parameter estimate using a KDE from the DataFrame of the results of the fit from the
    M2N object.

    :param data: (pandas.DataFrame) Contains parameter estimates from all runs.
    :param ignore_columns: (string, list) Column or columns to exclude from analysis.
    :param res: (int) Resolution of the kernel density estimate.
    :return: (dict) Labels and most likely estimates for parameters.
    """

    pass


@njit()
def iter_4_jit(mu_2, p_1, m_1, m_2, m_3, m_4):  # pragma: no cover
    """
    "Numbarized" evaluation of the set of equations that make up variant #1 of the EF3M algorithm (fitting using the
    first four moments).

    :param mu_2: (float) Initial parameter value for mu_2
    :param p_1: (float) Probability defining the mixture; p_1, 1 - p_1
    :param m_1, m_2, m_3, m_4: (float) The first four (1... 4) raw moments of the mixture distribution.
    :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,
        divide-by-zero), otherwise an empty list is returned.
    """

    pass


@njit()
def iter_5_jit(mu_2, p_1, m_1, m_2, m_3, m_4, m_5):  # pragma: no cover
    """
    "Numbarized" evaluation of the set of equations that make up variant #2 of the EF3M algorithm (fitting using the
     first five moments).

    :param mu_2: (float) Initial parameter value for mu_2
    :param p_1: (float) Probability defining the mixture; p_1, 1-p_1
    :param m_1, m_2, m_3, m_4, m_5: (float) The first five (1... 5) raw moments of the mixture distribution.
    :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,
        divide-by-zero), otherwise an empty list is returned.
    """

    pass
