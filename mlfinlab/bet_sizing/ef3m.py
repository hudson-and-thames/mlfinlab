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
        # Set fitting parameters in constructor.
        self.epsilon = epsilon
        self. factor = factor
        self.n_runs = n_runs
        self.variant = variant
        self.max_iter = max_iter
        self.num_workers = num_workers
        # Set moments to fit and initialize lists and errors.
        self.moments = moments
        self.new_moments = [0 for _ in range(5)]  # Initialize the new moment list to zeroes.
        self.parameters = [0 for _ in range(5)]  # Initialize the parameter list to zeroes.
        self.error = sum([moments[i]**2 for i in range(len(moments))])

    def fit(self, mu_2):
        """
        Fits and the parameters that describe the mixture of the 2 Normal distributions for a given set of initial
        parameter guesses.

        :param mu_2: (float) An initial estimate for the mean of the second distribution.
        """
        p_1 = np.random.uniform(0, 1)
        num_iter = 0
        while True:
            num_iter += 1
            if self.variant == 1:
                parameters_new = self.iter_4(mu_2, p_1)  # First variant, using the first 4 moments.
            elif self.variant == 2:
                parameters_new = self.iter_5(mu_2, p_1)  # Second variant, using all 5 moments.
            else:
                raise ValueError("Value of argument 'variant' must be either 1 or 2.")

            if not parameters_new:
                # An empty list returned means an invalid value was found in iter_4 or iter_5.
                return None

            parameters = parameters_new.copy()
            self.get_moments(parameters)
            error = sum([(self.moments[i] - self.new_moments[i])**2 for i in range(len(self.new_moments))])
            if error < self.error:
                # Update with new best parameters, error.
                self.parameters = parameters
                self.error = error

            if abs(p_1 - parameters[4]) < self.epsilon:
                # Stopping condition.
                break

            if num_iter > self.max_iter:
                # Stops calculation if algorithm reaches the set maximum number of iterations.
                return None

            p_1 = parameters[4]
            mu_2 = parameters[1]  # Update for the 5th moments convergence.

        self.parameters = parameters
        return None

    def get_moments(self, parameters, return_result=False):
        """
        Calculates and returns the first five (1...5) raw moments corresponding to the newly estimated parameters.

        :param parameters: (list) List of parameters if the specific order [mu_1, mu_2, sigma_1, sigma_2, p_1]
        :param return_result: (bool) If True, method returns a result instead of setting the 'self.new_moments'
         attribute.
        :return: (list) List of the first five moments
        """
        u_1, u_2, s_1, s_2, p_1 = parameters  # Expanded mixture parameters to individual variables for clarity.
        p_2 = 1 - p_1  # Explicitly state p_2 for symmetry.
        m_1 = p_1 * u_1 + p_2 * u_2  # Eq. (6)
        m_2 = p_1 * (s_1**2 + u_1**2) + p_2 * (s_2**2 + u_2**2)  # Eq. (7)
        m_3 = p_1 * (3 * s_1**2 * u_1 + u_1**3) + p_2 * (3 * s_2**2 * u_2 + u_2**3)  # Eq. (8)
        # Eq. (9)
        m_4 = p_1 * (3 * s_1**4 + 6 * s_1**2 * u_1**2 + u_1**4) + p_2 * (3 * s_2**4 + 6 * s_2**2 * u_2**2 + u_2**4)
        # Eq (10)
        m_5 = p_1 * (15 * s_1**4 * u_1 + 10 * s_1**2 * u_1**3 + u_1**5) + p_2 *\
            (15 * s_2**4 * u_2 + 10 * s_2**2 * u_2**3 + u_2**5)

        if return_result:
            return [m_1, m_2, m_3, m_4, m_5]

        self.new_moments = [m_1, m_2, m_3, m_4, m_5]
        return None

    def iter_4(self, mu_2, p_1):
        """
        Evaluation of the set of equations that make up variant #1 of the EF3M algorithm (fitting using the first
        four moments).

        :param mu_2: (float) Initial parameter value for mu_2
        :param p_1: (float) Probability defining the mixture; p_1, 1 - p_1
        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,
         divide-by-zero), otherwise an empty list is returned.
        """
        # Expand list of moments to individual variables for clarity.
        m_1, m_2, m_3, m_4 = self.moments[0:4]

        # Check to see if every value made it through.
        param_list = iter_4_jit(mu_2, p_1, m_1, m_2, m_3, m_4)
        param_list = param_list.tolist()

        if len(param_list) < 5:
            return []

        return param_list

    def iter_5(self, mu_2, p_1):
        """
        Evaluation of the set of equations that make up variant #2 of the EF3M algorithm (fitting using the first five
        moments).

        :param mu_2: (float) Initial parameter value for mu_2
        :param p_1: (float) Probability defining the mixture; p_1, 1-p_1
        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,
         divide-by-zero), otherwise an empty list is returned.
        """

        # Expand list of moments to individual variables for clarity.
        (m_1, m_2, m_3, m_4, m_5,) = self.moments

        # Call numba decorated function to do the actual calculations
        param_list = iter_5_jit(mu_2, p_1, m_1, m_2, m_3, m_4, m_5)

        param_list = param_list.tolist()

        if len(param_list) < 5:
            return []

        return param_list

    def single_fit_loop(self, epsilon=0):
        """
        A single scan through the list of mu_2 values, cataloging the successful fittings in a DataFrame.

        :param epsilon: (float) Fitting tolerance.
        :return: (pd.DataFrame) Fitted parameters and error
        """
        # Reset parameters and error for each single_fit_loop.
        self.epsilon = epsilon if epsilon != 0 else self.epsilon
        self.parameters = [0 for _ in range(5)]  # Initialize the parameter list.
        self.error = sum([self.moments[i]**2 for i in range(len(self.moments))])

        std_dev = centered_moment(self.moments, 2)**0.5
        mu_2 = [float(i) * self.epsilon * self.factor * std_dev + self.moments[0] for i in range(1, int(1/self.epsilon))]
        err_min = self.error

        d_results = {}
        for mu_2_i in mu_2:
            self.fit(mu_2=mu_2_i)

            if self.error < err_min:
                err_min = self.error
                d_results['mu_1'], d_results['mu_2'], d_results['sigma_1'], d_results['sigma_2'], d_results['p_1'] = \
                    [[p] for p in self.parameters]
                d_results['error'] = [err_min]

        return pd.DataFrame.from_dict(d_results)

    def mp_fit(self):
        """
        Parallelized implementation of the 'single_fit_loop' method. Makes use of dask.delayed to execute multiple
        calls of 'single_fit_loop' in parallel.

        :return: (pd.DataFrame) Fitted parameters and error
        """
        num_workers = self.num_workers if self.num_workers > 0 else cpu_count()
        pool = Pool(num_workers)

        output_list = pool.imap_unordered(self.single_fit_loop, [self.epsilon for i in range(self.n_runs)])
        df_list = []

        # Process asynchronous output, report progress and progress bar.
        max_prog_bar_len = 25
        for i, out_i in enumerate(output_list, 1):
            df_list.append(out_i)
            num_fill = int((i/self.n_runs) * max_prog_bar_len)
            prog_bar_string = '|' + num_fill*'#' + (max_prog_bar_len-num_fill)*' ' + '|'
            sys.stderr.write(f'\r{prog_bar_string} Completed {i} of {self.n_runs} fitting rounds.')
        # Close and clean up pool.
        pool.close()
        pool.join()
        # Concatenate and return results of fitting.
        df_out = pd.concat(df_list)
        return df_out


# === Helper functions, outside the M2N class. === #
def centered_moment(moments, order):
    """
    Compute a single moment of a specific order about the mean (centered) given moments about the origin (raw).

    :param moments: (list) First 'order' raw moments
    :param order: (int) The order of the moment to calculate
    :return: (float) The central moment of specified order.
    """
    moment_c = 0  # First centered moment is always zero.
    for j in range(order + 1):
        combin = int(comb(order, j))
        if j == order:
            a_1 = 1
        else:
            a_1 = moments[order - j - 1]
        moment_c += (-1)**j * combin * moments[0]**j * a_1

    return moment_c


def raw_moment(central_moments, dist_mean):
    """
    Calculates a list of raw moments given a list of central moments.

    :param central_moments: (list) The first n (1...n) central moments as a list.
    :param dist_mean: (float) The mean of the distribution.
    :return: (list) The first n+1 (0...n) raw moments.
    """
    raw_moments = [dist_mean]
    central_moments = [1] + central_moments  # Add the zeroth moment to the front of the list, just helps with indexing.
    for n_i in range(2, len(central_moments)):
        moment_n_parts = []
        for k in range(n_i + 1):
            sum_part = comb(n_i, k) * central_moments[k] * dist_mean**(n_i - k)
            moment_n_parts.append(sum_part)
        moment_n = sum(moment_n_parts)
        raw_moments.append(moment_n)
    return raw_moments


def most_likely_parameters(data, ignore_columns='error', res=10_000):
    """
    Determines the most likely parameter estimate using a KDE from the DataFrame of the results of the fit from the
    M2N object.

    :param data: (pandas.DataFrame) Contains parameter estimates from all runs.
    :param ignore_columns: (string, list) Column or columns to exclude from analysis.
    :param res: (int) Resolution of the kernel density estimate.
    :return: (dict) Labels and most likely estimates for parameters.
    """
    df_results = data.copy()
    if isinstance(ignore_columns, str):
        ignore_columns = [ignore_columns]

    columns = [c for c in df_results.columns if c not in ignore_columns]
    d_results = {}
    for col in columns:
        x_range = np.linspace(df_results[col].min(), df_results[col].max(), num=res)
        kde = gaussian_kde(df_results[col].to_numpy())
        y_kde = kde.evaluate(x_range)
        top_value = round(x_range[np.argmax(y_kde)], 5)
        d_results[col] = top_value

    return d_results


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
    param_list = np.empty(0, dtype=np.float64)

    # Using a while-loop here to be able to use 'break' functionality.
    # We need to stop the calculation at any given step to avoid throwing warnings or errors,
    # and be in control of our return values. I'm open to other suggestions, but multiple return statements isn't
    # one of them.
    while True:
        # Calculate mu_1, Equation (22).
        mu_1 = (m_1 - (1 - p_1) * mu_2) / p_1

        # Calculate sigma_2, Equation (24)
        if (3 * (1 - p_1) * (mu_2 - mu_1)) == 0:
            # Validity check 1: Check for divide-by-zero.
            break

        sigma_2_squared = (
            m_3
            + 2 * p_1 * mu_1 ** 3
            + (p_1 - 1) * mu_2 ** 3
            - 3 * mu_1 * (m_2 + mu_2 ** 2 * (p_1 - 1))
        ) / (3 * (1 - p_1) * (mu_2 - mu_1))
        if sigma_2_squared < 0:
            # Validity check 2: Prevent potential complex values.
            break

        sigma_2 = sigma_2_squared ** 0.5
        # Calculate sigma_1, Equation (23)
        sigma_1_squared = (
            (m_2 - sigma_2 ** 2 - mu_2 ** 2) / p_1
            + sigma_2 ** 2
            + mu_2 ** 2
            - mu_1 ** 2
        )

        if sigma_1_squared < 0:
            # Validity check 3: Prevent potential complex values.
            break
        sigma_1 = sigma_1_squared ** 0.5

        # Adjust guess for p_1, Equation (25)
        p_1_deno = (
            3 * (sigma_1 ** 4 - sigma_2 ** 4)
            + 6 * (sigma_1 ** 2 * mu_1 ** 2 - sigma_2 ** 2 * mu_2 ** 2)
            + mu_1 ** 4
            - mu_2 ** 4
        )
        if p_1_deno == 0:
            # Validity check 5: Break if about to divide by zero.
            break

        p_1 = (
            m_4 - 3 * sigma_2 ** 4 - 6 * sigma_2 ** 2 * mu_2 ** 2 - mu_2 ** 4
        ) / p_1_deno
        if (p_1 < 0) or (p_1 > 1):
            # Validity check 6: The probability must be between zero and one.
            break

        # Add all new parameter estimates to the return list if no break has occurred before now.
        param_list = np.array([mu_1, mu_2, sigma_1, sigma_2, p_1], dtype=np.float64)

        # We only want this to execute once at most, so call a final break if one hasn't been called yet.
        break

    return param_list


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
    param_list = np.empty(0, dtype=np.float64)

    # Using a while-loop here to be able to use 'break' functionality.
    # We need to stop the calculation at any given step to avoid throwing warnings or errors, and be in control
    # of our return values. I'm open to other suggestions, but multiple return statements isn't one of them.
    while True:
        # Calculate mu_1, Equation (22).
        mu_1 = (m_1 - (1 - p_1) * mu_2) / p_1
        if (3 * (1 - p_1) * (mu_2 - mu_1)) == 0:
            # Validity check 1: check for divide-by-zero.
            break

        # Calculate sigma_2, Equation (24).
        sigma_2_squared = (m_3 + 2 * p_1 * mu_1 ** 3 + (p_1 - 1) * mu_2 ** 3 - 3 * mu_1 * (m_2 + mu_2 ** 2 * (p_1 - 1))
                           ) / (3 * (1 - p_1) * (mu_2 - mu_1))

        if sigma_2_squared < 0:
            # Validity check 2: check for upcoming complex numbers.
            break

        sigma_2 = sigma_2_squared ** 0.5

        # Calculate sigma_1, Equation (23).
        sigma_1_squared = ((m_2 - sigma_2 ** 2 - mu_2 ** 2) / p_1 + sigma_2 ** 2 + mu_2 ** 2 - mu_1 ** 2)
        if sigma_1_squared < 0:
            # Validity check 3: check for upcoming complex numbers.
            break

        sigma_1 = sigma_1_squared ** 0.5

        # Adjust the guess for mu_2, Equation (27).
        if (1 - p_1) < 1e-4:
            # Validity check 5: break to prevent divide-by-zero.
            break

        a_1_squared = 6 * sigma_2 ** 4 + (m_4 - p_1 * (3 * sigma_1 ** 4 + 6 * sigma_1 ** 2 * mu_1 ** 2 + mu_1 ** 4)
                                          ) / (1 - p_1)
        if a_1_squared < 0:
            # Validity check 6: break to avoid taking the square root of negative number.
            break

        a_1 = a_1_squared ** 0.5
        mu_2_squared = a_1 - 3 * sigma_2 ** 2

        # Validity check 7: break to avoid complex numbers.
        # Todo: Avoid Numba object mode.
        # Numba does not support numpy.iscomplex. This creates an overhead.
        with objmode(mu_2_squared_is_complex="boolean"):
            mu_2_squared_is_complex = bool(np.iscomplex(mu_2_squared))
        if mu_2_squared_is_complex or mu_2_squared < 0:
            break

        mu_2 = mu_2_squared ** 0.5
        # Adjust guess for p_1, Equation (28, 29).
        a_2 = 15 * sigma_1 ** 4 * mu_1 + 10 * sigma_1 ** 2 * mu_1 ** 3 + mu_1 ** 5
        b_2 = 15 * sigma_2 ** 4 * mu_2 + 10 * sigma_2 ** 2 * mu_2 ** 3 + mu_2 ** 5
        if (a_2 - b_2) == 0:
            # Validity check 8: break to prevent divide-by-zero.
            break

        p_1 = (m_5 - b_2) / (a_2 - b_2)
        if (p_1 < 0) or (p_1 > 1):
            # Validity check 9: p_1 value must be between 0 and 1.
            break

        # Add all new parameter estimates to the return list if no break has occurred before now.
        param_list = np.array([mu_1, mu_2, sigma_1, sigma_2, p_1], dtype=np.float64)

        # We only want this to execute once at most, so call a final break if one hasn't been called yet.
        break

    return param_list
