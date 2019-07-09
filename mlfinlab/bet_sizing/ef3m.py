"""
An implementation of the Exact Fit of the first 3 Moments
(#F3M) of finding the parameters that make up the mixture
of 2 Gaussian distributions. Based on the work by
Lopez de Prado and Foreman (2014) "A mixture of two
Gaussians approach to mathematical portfolio oversight:
The EF3M algorithm." Quantitative Finance, Vol. 14,
No. 5, pp. 913-930.
"""

# imports
import numpy as np

from scipy.special import comb

import pandas as pd
import dask.dataframe as dd
from dask import delayed


class M2N:
    """
    M2N - A Mixture of 2 Normal distributions
    This class is used to contain parameters and equations for the
    EF3M algorithm, when fitting parameters to a mixture
    of 2 Gaussians.
    """
    def __init__(self, moments):
        """
        Constructor

        :param moments: (list) The first five (1... 5) raw moments of the
                        mixture distribution.

        The parameters of the mixture are defined by a list, where:
        parameters = [mu1, mu2, sigma1, sigma2, p1]
        """
        self.moments = moments
        self.parameters = [0 for i in range(5)]  # initialize parameter list
        self.error = sum([moments[i]**2 for i in range(len(moments))])

    def fit(self, mu2, epsilon, variant=1, maxIter=100_000):
        """
        Fits and the parameters that describe the mixture
        of the 2 Normal distributions for a given set of initial
        parameter guesses.

        :param mu2: (float) An initial estimate for the mean of the second
                    distribution.
        :param epsilon: (float) Error tolerance.
        :param variant: (int) Which algorithm variant to use, 1 or 2.
        :param maxIter: (int) Maximum number of iterations after which
                        to terminate loop.
        """
        p1 = np.random.uniform(0, 1)
        numIter = 0
        while True:
            numIter += 1
            if variant == 1:
                parameters_new = self.iter4(mu2, p1, self.moments)  # first variant
            elif variant == 2:
                parameters_new = self.iter5(mu2, p1, self.moments)  # second variant
            else:
                raise ValueError("Value of 'variant' must be either 1 or 2.")
            if len(parameters_new) == 0:
                # invalid value found in iter4 or iter5
                return None
            parameters = parameters_new.copy()
            moments = self.get_moments(parameters)
            error = sum([(self.moments[i]-moments[i])**2
                         for i in range(len(moments))])
            if error < self.error:
                # update with new best parameters, error
                self.parameters = parameters
                self.error = error
            if abs(p1 - parameters[4]) < epsilon:
                # stopping condition
                break
            if numIter > maxIter:
                # maxIter reached, convergence not fast enough
                return None
            p1 = parameters[4]
            mu2 = parameters[1]  # for the 5th moments convergence
        self.parameters = parameters
        return None

    def get_moments(self, parameters):
        """
        Calculates and returns the first five (1...5) raw moments
        corresponding to the newly esitmated parameters.

        :param parameters: (list) List of parameters if the
        specific order [mu1, mu2, sigma1, sigma2, p1]
        :return: (list) List of the first five moments
        """
        u1, u2, s1, s2, p1 = parameters  # for clarity
        p2 = 1-p1  # for symmetry
        m1 = p1*u1 + p2*u2  # Eq. (6)
        m2 = p1*(s1**2 + u1**2) + p2*(s2**2 + u2**2)  # Eq. (7)
        m3 = p1*(3*s1**2*u1 + u1**3) + p2*(3*s2**2*u2 + u2**3)  # Eq. (8)
        m4 = p1*(3*s1**4 + 6*s1**2*u1**2 + u1**4) +\
            p2*(3*s2**4 + 6*s2**2*u2**2 + u2**4)  # Eq. (9)
        m5 = p1*(15*s1**4*u1 + 10*s1**2*u1**3 + u1**5) +\
            p2*(15*s2**4*u2 + 10*s2**2*u2**3 + u2**5)  # Eq. (10)
        return [m1, m2, m3, m4, m5]

    def iter4(self, mu2, p1, moments):
        """
        Evaluation of the set of equations that make up
        variant #1 of the EF3M algorithm (fitting using the
        first four moments).

        :param mu2: (float) Initial parameter value for mu2
        :param p1: (float) Probability defining the mixture; p1, 1-p1
        :param moments: (list) First five raw moments of the mixture
        distribution [m1, m2, m3, m4, m5]
        :return: (list) List of estimated parameter if no invalid values
        are encountered (e.g. complex values, divide-by-zero), otherwise
        an empty list is returned.
        """
        m1, m2, m3, m4 = moments[0:4]  # for clarity
        # mu1, Equation (22)
        mu1 = (m1 - (1-p1)*mu2) / p1
        # sigma2, Equation (24)
        if (3*(1-p1)*(mu2-mu1)) == 0:
            # check for divide-by-zero
            return []
        sigma2_squared = ( (m3 + 2*p1*mu1**3 + (p1-1)*mu2**3 -\
                                        3*mu1*(m2 + mu2**2*(p1-1))) /\
                            (3*(1-p1)*(mu2-mu1)) )
        if sigma2_squared < 0:
            return []
        sigma2 = sigma2_squared**(.5)
        # sigma1, Equation (23)
        sigma1_squared = ( (m2 - sigma2**2 - mu2**2)/p1 +\
                                        sigma2**2 + mu2**2 - mu1**2 )
        if sigma1_squared < 0:
            return []
        sigma1 = sigma1_squared**(.5)
        if np.iscomplex(sigma1) or np.iscomplex(sigma2) or \
            np.isnan(sigma1) or np.isnan(sigma2):
            return []  # returns empty list sigma1 or sigma2 are invalid
        # adjust guess for p1, Equation (25)
        p1_deno = (3*(sigma1**4 - sigma2**4) + 6*(sigma1**2*mu1**2 - \
            sigma2**2*mu2**2) + mu1**4 - mu2**4)
        if p1_deno == 0:
            return []  # return empty list if about to divide by zero
        p1 = (m4 - 3*sigma2**4 - 6*sigma2**2*mu2**2 - mu2**4) / p1_deno
        if (p1<0) or (p1>1):
            return []
        return [mu1, mu2, sigma1, sigma2, p1]
    
    def iter5(self, mu2, p1, moments):
        """
        Evaluation of the set of equations that make up
        variant #2 of the EF3M algorithm (fitting using the 
        first five moments).

        :param mu2: (float) Initial parameter value for mu2
        :param p1: (float) Probability defining the mixture; p1, 1-p1
        :param moments: (list) First five raw moments of the mixture
        distribution [m1, m2, m3, m4, m5]
        :return: (list) List of estimated parameter if no invalid values
        are encountered (e.g. complex values, divide-by-zero), otherwise
        an empty list is returned.
        """
        m1, m2, m3, m4, m5 = moments  # for clarity
        # mu1, Equation (22)
        mu1 = (m1 - (1-p1)*mu2) / p1
        if (3*(1-p1)*(mu2-mu1)) == 0:
            return []
        # sigma2, Equation (24)
        if (3*(1-p1)*(mu2-mu1)) == 0:
            # check for divide-by-zero
            return []
        sigma2_squared = ( (m3 + 2*p1*mu1**3 + (p1-1)*mu2**3 -\
                                        3*mu1*(m2 + mu2**2*(p1-1))) /\
                            (3*(1-p1)*(mu2-mu1)) )
        if sigma2_squared < 0:
            return []
        sigma2 = sigma2_squared**(.5)
        # sigma1, Equation (23)
        sigma1_squared = ( (m2 - sigma2**2 - mu2**2)/p1 +\
                                        sigma2**2 + mu2**2 - mu1**2 )
        if sigma1_squared < 0:
            return []
        sigma1 = sigma1_squared**(.5)
        # last check for sigma1 and sigma2 validity
        if np.iscomplex(sigma1) or np.iscomplex(sigma2) or \
            np.isnan(sigma1) or np.isnan(sigma2):
            return []
        # adjust the guess for mu2, Equation (27)
        if (1-p1) < 1e-4:
            return []
        a = ( 6*sigma2**4 + (m4-p1*(3*sigma1**4+6*sigma1**2*mu1**2+mu1**4)) /\
             (1-p1 ) )**.5
        mu2_squared = (a - 3*sigma2**2)
        if np.iscomplex(mu2_squared):
            return []
        if mu2_squared < 0:
            return []
        mu2 = mu2_squared**.5
        if np.iscomplex(mu2):
            return []
        # adjust guess for p1, Equation (28, 29)
        a = 15*sigma1**4*mu1+10*sigma1**2*mu1**3+mu1**5
        b = 15*sigma2**4*mu2+10*sigma2**2*mu2**3+mu2**5
        if (a-b) == 0:
            return []  # return empty list if about to divide by zero
        p1 = (m5-b) / (a-b)
        if (p1<0) or (p1>1):
            return []
        return [mu1, mu2, sigma1, sigma2, p1]

    def singleLoop(self, moments, epsilon=10**-5, factor=5,
                    variant=1, maxIter=100_000):
        """
        A single scan through the list of mu2 values, cataloging the
        successful fittings in a DataFrame.

        :param moments: (list) First five central moments, [m1, m2, m3, m4, m5]
        :param epsilon: (float) Fitting tolerance
        :param factor: (float) Lambda factor from equations
        :param variant: (int) The EF3M variant to execute, options
        are 1: EF3M using first 4 moments, 2: EF3M using first 5 moments
        :param maxIter: (int) Maximum number of iterations to perform
        in the 'fit' method
        :return: (pd.DataFrame) Fitted parameters and error
        """
        stDev = centeredMoment(moments, 2)**.5
        mu2 = [float(i)*epsilon*factor*stDev + moments[0] 
                for i in range(1, int(1/epsilon))]
        m2n = M2N(moments)
        err_min = m2n.error
        d_results = {}
        for mu2_i in mu2:
            m2n.fit(mu2=mu2_i, epsilon=epsilon, variant=variant, maxIter=maxIter)
            if m2n.error < err_min:
                err_min = m2n.error
                d_results['mu1'], d_results['mu2'], d_results['sigma1'], \
                    d_results['sigma2'], \
                        d_results['p1'] = [[p] for p in m2n.parameters]
                d_results['error'] = [err_min]
        return pd.DataFrame.from_dict(d_results)

    # repeat runs and collect results as a DataFrame
    def mpFit(self, moments, epsilon=10**-5, factor=5, n_runs=1, variant=1,
                maxIter=100_000):
        """
        Parallelized implementation of 'singleLoop' method.

        :param moments: (list) First five central moments, [m1, m2, m3, m4, m5]
        :param epsilon: (float) Fitting tolerance
        :param factor: (float) Lambda factor from equations
        :param n_runs: (int) Number of times to execute 'singleLoop'
        :param variant: (int) The EF3M variant to execute, options
        are 1: EF3M using first 4 moments, 2: EF3M using first 5 moments
        :param maxIter: (int) Maximum number of iterations to perform
        in the 'fit' method
        :return: (pd.DataFrame) Fitted parameters and error
        """
        # create a list of delayed objects that return a pd.DataFrame
        dfs = [delayed(self.singleLoop)(moments=moments,
                                        epsilon=epsilon,
                                        factor=factor,
                                        variant=variant,
                                        maxIter=maxIter
                                        ) for i in range(n_runs)]
        # build a dask.DataFrame from a list of delayed objects
        ddf = dd.from_delayed(dfs)
        # compute all runs, using dask multiprocessing
        df = ddf.compute(scheduler='processes').reset_index(drop=True)
        df = df.sort_values('error')
        return df


# === Helper functions, outside class === #
def centeredMoment(moments, order):
    """
    Compute a single moment of a specific order about the mean (centered)
    given moments about the origin (raw).

    :param moments: (list) First 'order' raw moments
    :param order: (int) The order of the moment to calculate
    """
    moment_c = 0  # first centered moment is always zero
    for j in range(order + 1):
        combin = binomialCoeff(order, j)
        if j == order:
            a = 1
        else:
            a = moments[order-j-1]
        moment_c += (-1)**j*combin*moments[0]**j*a
    return moment_c


# calculate raw moments from moments about the mean (central moments)
def rawMoment(central_moments, dist_mean):
    """
    Calculates a list of raw moments given a list of 
    central moments.


    :param central_moments: (list) The first n (1...n) central moments as a list
    :param dist_mean: (float) The mean of the distribution
    :return: (list) The first n+1 (0...n) raw moments 
    """
    raw_moments = [dist_mean]
    central_moments = [1] + central_moments  # add the zeroth moment
    for n in range(2, len(central_moments)):
        moment_n_parts = []
        for k in range(n+1):
            sum_part = comb(n, k) * central_moments[k] * dist_mean**(n-k)
            moment_n_parts.append(sum_part)
        moment_n = sum(moment_n_parts)
        raw_moments.append(moment_n)
    return raw_moments


# number of combinations of n over k
def binomialCoeff(n, k):
    """
    Calculate the number of way 'n' things can be chosen 'k' at-a-time,
    'n'-choose-'k'. This is a simple implementation of the
    scipy.special.comb function.

    :param n: (int) The number of things
    :param k: (int) The number of things to be chosen at one time
    :return: (int) The total number of combinations
    """
    if k < 0 or k > n:
        return 0
    if k > n-k:
        k = n-k
    c = 1
    for i in range(k):
        c = c*(n - (k - (i+1)))
        c = c // (i+1)
    return c
