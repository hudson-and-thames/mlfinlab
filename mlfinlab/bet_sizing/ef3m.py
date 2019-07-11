"""
An implementation of the Exact Fit of the first 3 Moments (#F3M) of finding the parameters that make up the mixture
of 2 Gaussian distributions. Based on the work by Lopez de Prado and Foreman (2014) "A mixture of two Gaussians approach to
mathematical portfolio oversight: The EF3M algorithm." Quantitative Finance, Vol. 14, No. 5, pp. 913-930.
"""

# imports
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from scipy.special import comb


class M2N:
    """
    m2n - A Mixture of 2 Normal distributions
    This class is used to contain parameters and equations for the EF3M algorithm, when fitting parameters to a mixture of 2 Gaussians.
    """
    def __init__(self, moments):
        """
        Constructor

        :param moments: (list) The first five (1... 5) raw moments of the mixture distribution.

        The parameters of the mixture are defined by a list, where:
            parameters = [mu_1, mu_2, sigma_1, sigma_2, p_1]
        """
        self.moments = moments
        self.parameters = [0 for i in range(5)]  # initialize parameter list
        self.error = sum([moments[i]**2 for i in range(len(moments))])

    def fit(self, mu_2, epsilon, variant=1, max_iter=100_000):
        """
        Fits and the parameters that describe the mixture of the 2 Normal distributions for a given set of initial parameter guesses.

        :param mu_2: (float) An initial estimate for the mean of the second distribution.
        :param epsilon: (float) Error tolerance.
        :param variant: (int) Which algorithm variant to use, 1 or 2.
        :param max_iter: (int) Maximum number of iterations after which to terminate loop.
        """
        p_1 = np.random.uniform(0, 1)
        num_iter = 0
        while True:
            num_iter += 1
            if variant == 1:
                parameters_new = self.iter_4(mu_2, p_1, self.moments)  # first variant
            elif variant == 2:
                parameters_new = self.iter_5(mu_2, p_1, self.moments)  # second variant
            else:
                raise ValueError("Value of 'variant' must be either 1 or 2.")
            if not parameters_new:
                # An empty list returned means an invalid value was found in iter4 or iter5.
                return None
            parameters = parameters_new.copy()
            moments = self.get_moments(parameters)
            error = sum([(self.moments[i]-moments[i])**2 for i in range(len(moments))])
            if error < self.error:
                # update with new best parameters, error
                self.parameters = parameters
                self.error = error
            if abs(p_1 - parameters[4]) < epsilon:
                # stopping condition
                break
            if num_iter > max_iter:
                # max_iter reached, convergence not fast enough
                return None
            p_1 = parameters[4]
            mu_2 = parameters[1]  # for the 5th moments convergence
        self.parameters = parameters
        return None

    def get_moments(self, parameters):
        """
        Calculates and returns the first five (1...5) raw moments corresponding to the newly esitmated parameters.

        :param parameters: (list) List of parameters if the specific order [mu_1, mu_2, sigma_1, sigma_2, p_1]
        :return: (list) List of the first five moments
        """
        u_1, u_2, s_1, s_2, p_1 = parameters  # for clarity
        p_2 = 1-p_1  # for symmetry
        m_1 = p_1*u_1 + p_2*u_2  # Eq. (6)
        m_2 = p_1*(s_1**2 + u_1**2) + p_2*(s_2**2 + u_2**2)  # Eq. (7)
        m_3 = p_1*(3*s_1**2*u_1 + u_1**3) + p_2*(3*s_2**2*u_2 + u_2**3)  # Eq. (8)
        m_4 = p_1*(3*s_1**4 + 6*s_1**2*u_1**2 + u_1**4) + p_2*(3*s_2**4 + 6*s_2**2*u_2**2 + u_2**4)  # Eq. (9)
        m_5 = p_1*(15*s_1**4*u_1 + 10*s_1**2*u_1**3 + u_1**5) + p_2*(15*s_2**4*u_2 + 10*s_2**2*u_2**3 + u_2**5)  # Eq. (10)
        return [m_1, m_2, m_3, m_4, m_5]

    def iter_4(self, mu_2, p_1, moments):
        """
        Evaluation of the set of equations that make up variant #1 of the EF3M algorithm (fitting using the first four moments).

        :param mu_2: (float) Initial parameter value for mu_2
        :param p_1: (float) Probability defining the mixture; p_1, 1-p_1
        :param moments: (list) First five raw moments of the mixture distribution [m_1, m_2, m_3, m_4, m_5]
        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values, divide-by-zero), otherwise
         an empty list is returned.
        """
        m_1, m_2, m_3, m_4 = moments[0:4]  # for clarity
        # mu_1, Equation (22)
        mu_1 = (m_1 - (1-p_1)*mu_2) / p_1
        # sigma_2, Equation (24)
        if (3*(1-p_1)*(mu_2-mu_1)) == 0:
            # check for divide-by-zero
            return []
        sigma_2_squared = ((m_3 + 2*p_1*mu_1**3 + (p_1-1)*mu_2**3 - 3*mu_1*(m_2 + mu_2**2*(p_1-1))) / (3*(1-p_1)*(mu_2-mu_1)))
        if sigma_2_squared < 0:
            return []
        sigma_2 = sigma_2_squared**(.5)
        # sigma_1, Equation (23)
        sigma_1_squared = ((m_2 - sigma_2**2 - mu_2**2)/p_1 + sigma_2**2 + mu_2**2 - mu_1**2)
        if sigma_1_squared < 0:
            return []
        sigma_1 = sigma_1_squared**(.5)
        if np.iscomplex(sigma_1) or np.iscomplex(sigma_2) or \
            np.isnan(sigma_1) or np.isnan(sigma_2):
            return []  # returns empty list sigma_1 or sigma_2 are invalid
        # adjust guess for p_1, Equation (25)
        p_1_deno = (3*(sigma_1**4 - sigma_2**4) + 6*(sigma_1**2*mu_1**2 - sigma_2**2*mu_2**2) + mu_1**4 - mu_2**4)
        if p_1_deno == 0:
            return []  # return empty list if about to divide by zero
        p_1 = (m_4 - 3*sigma_2**4 - 6*sigma_2**2*mu_2**2 - mu_2**4) / p_1_deno
        if (p_1 < 0) or (p_1 > 1):
            return []
        return [mu_1, mu_2, sigma_1, sigma_2, p_1]

    def iter_5(self, mu_2, p_1, moments):
        """
        Evaluation of the set of equations that make up variant #2 of the EF3M algorithm (fitting using the first five moments).

        :param mu_2: (float) Initial parameter value for mu_2
        :param p_1: (float) Probability defining the mixture; p_1, 1-p_1
        :param moments: (list) First five raw moments of the mixture distribution [m_1, m_2, m_3, m_4, m_5]
        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values, divide-by-zero), otherwise
         an empty list is returned.
        """
        m_1, m_2, m_3, m_4, m_5 = moments  # for clarity
        # mu_1, Equation (22)
        mu_1 = (m_1 - (1-p_1)*mu_2) / p_1
        if (3*(1-p_1)*(mu_2-mu_1)) == 0:
            return []
        # sigma_2, Equation (24)
        if (3*(1-p_1)*(mu_2-mu_1)) == 0:
            # check for divide-by-zero
            return []
        sigma_2_squared = ((m_3 + 2*p_1*mu_1**3 + (p_1-1)*mu_2**3 - 3*mu_1*(m_2 + mu_2**2*(p_1-1))) / (3*(1-p_1)*(mu_2-mu_1)))
        if sigma_2_squared < 0:
            return []
        sigma_2 = sigma_2_squared**(.5)
        # sigma_1, Equation (23)
        sigma_1_squared = ((m_2 - sigma_2**2 - mu_2**2)/p_1 + sigma_2**2 + mu_2**2 - mu_1**2)
        if sigma_1_squared < 0:
            return []
        sigma_1 = sigma_1_squared**(.5)
        # last check for sigma_1 and sigma_2 validity
        if np.iscomplex(sigma_1) or np.iscomplex(sigma_2) or np.isnan(sigma_1) or np.isnan(sigma_2):
            return []
        # adjust the guess for mu_2, Equation (27)
        if (1-p_1) < 1e-4:
            return []
        a_1 = (6*sigma_2**4 + (m_4-p_1*(3*sigma_1**4+6*sigma_1**2*mu_1**2+mu_1**4)) / (1-p_1))**.5
        mu_2_squared = (a_1 - 3*sigma_2**2)
        if np.iscomplex(mu_2_squared) or mu_2 < 0:
            return []
        #if mu_2_squared < 0:
        #    return []
        mu_2 = mu_2_squared**.5
        if np.iscomplex(mu_2):
            return []
        # adjust guess for p_1, Equation (28, 29)
        a_2 = 15*sigma_1**4*mu_1+10*sigma_1**2*mu_1**3+mu_1**5
        b_2 = 15*sigma_2**4*mu_2+10*sigma_2**2*mu_2**3+mu_2**5
        if (a_2-b_2) == 0:
            return []  # return empty list if about to divide by zero
        p_1 = (m_5-b_2) / (a_2-b_2)
        if (p_1 < 0) or (p_1 > 1):
            return []
        return [mu_1, mu_2, sigma_1, sigma_2, p_1]

    def single_fit_loop(self, epsilon=10**-5, factor=5, variant=1, max_iter=100_000):
        """
        A single scan through the list of mu_2 values, cataloging the successful fittings in a DataFrame.

        :param moments: (list) First five central moments, [m_1, m_2, m_3, m_4, m_5]
        :param epsilon: (float) Fitting tolerance
        :param factor: (float) Lambda factor from equations
        :param variant: (int) The EF3M variant to execute, options are 1: EF3M using first 4 moments, 2: EF3M using first 5 moments
        :param max_iter: (int) Maximum number of iterations to perform in the 'fit' method
        :return: (pd.DataFrame) Fitted parameters and error
        """
        # Reset parameters and error for each single_fit_loop.
        self.parameters = [0 for i in range(5)]  # initialize parameter list
        self.error = sum([self.moments[i]**2 for i in range(len(self.moments))])

        std_dev = centered_moment(self.moments, 2)**.5
        mu_2 = [float(i)*epsilon*factor*std_dev + self.moments[0] for i in range(1, int(1/epsilon))]
        err_min = self.error
        d_results = {}
        for mu_2_i in mu_2:
            self.fit(mu_2=mu_2_i, epsilon=epsilon, variant=variant, max_iter=max_iter)
            if self.error < err_min:
                err_min = self.error
                d_results['mu_1'], d_results['mu_2'], d_results['sigma_1'], d_results['sigma_2'], d_results['p_1'] = [[p] for p in self.parameters]
                d_results['error'] = [err_min]
        return pd.DataFrame.from_dict(d_results)

    # repeat runs and collect results as a DataFrame
    def mp_fit(self, epsilon=10**-5, factor=5, n_runs=1, variant=1, max_iter=100_000, num_workers=-1):
        """
        Parallelized implementation of the 'single_fit_loop' method. Makes use of dask.delayed

        :param moments: (list) First five central moments, [m_1, m_2, m_3, m_4, m_5]
        :param epsilon: (float) Fitting tolerance
        :param factor: (float) Lambda factor from equations
        :param n_runs: (int) Number of times to execute 'singleLoop'
        :param variant: (int) The EF3M variant to execute, options are 1: EF3M using first 4 moments, 2: EF3M using first 5 moments
        :param max_iter: (int) Maximum number of iterations to perform in the 'fit' method
        :param num_workers: (int) Number of CPU cores to use for multiprocessing execution. Default is -1 which sets num_workers to all cores.
        :return: (pd.DataFrame) Fitted parameters and error
        """
        # create a list of delayed objects that return a pd.DataFrame
        dfs = [delayed(self.single_fit_loop)(epsilon=epsilon, factor=factor, variant=variant, max_iter=max_iter) for i in range(n_runs)]
        # build a dask.DataFrame from a list of delayed objects
        ddf = dd.from_delayed(dfs)
        # compute all runs, using dask multiprocessing
        num_workers = num_workers if num_workers > 0 else cpu_count()
        df_out = ddf.compute(scheduler='processes', num_workers=num_workers).reset_index(drop=True)
        df_out = df_out.sort_values('error')
        return df_out

# === Helper functions, outside the m2n class. === #
def centered_moment(moments, order):
    """
    Compute a single moment of a specific order about the mean (centered) given moments about the origin (raw).

    :param moments: (list) First 'order' raw moments
    :param order: (int) The order of the moment to calculate
    """
    moment_c = 0  # first centered moment is always zero
    for j in range(order + 1):
        combin = int(comb(order, j))
        if j == order:
            a_1 = 1
        else:
            a_1 = moments[order-j-1]
        moment_c += (-1)**j*combin*moments[0]**j*a_1
    return moment_c


def raw_moment(central_moments, dist_mean):
    """
    Calculates a list of raw moments given a list of central moments.

    :param central_moments: (list) The first n (1...n) central moments as a list.
    :param dist_mean: (float) The mean of the distribution.
    :return: (list) The first n+1 (0...n) raw moments.
    """
    raw_moments = [dist_mean]
    central_moments = [1] + central_moments  # add the zeroth moment
    for n_i in range(2, len(central_moments)):
        moment_n_parts = []
        for k in range(n_i+1):
            sum_part = comb(n_i, k) * central_moments[k] * dist_mean**(n_i-k)
            moment_n_parts.append(sum_part)
        moment_n = sum(moment_n_parts)
        raw_moments.append(moment_n)
    return raw_moments
