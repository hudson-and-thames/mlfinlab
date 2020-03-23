# pylint: disable=missing-module-docstring
import numpy as np
import scipy.stats as ss
from scipy import linalg



class CampbellBacktesting:
    """
    This class implements the Haircut Sharpe Ratios and Profit Hurdles algorithms described in the following paper:
    `Campbell R. Harvey and Yan Liu, Backtesting, (Fall 2015). Journal of Portfolio Management,
    2015 <https://papers.ssrn.com/abstract_id=2345489>`_; The code is based on the code provided by the authors of the paper.

    The Haircut Sharpe Ratios algorithm lets the user adjust the observed Sharpe Ratios to take multiple testing into account
    and calculate the corresponding haircuts. The haircut is the percentage difference between the original Sharpe ratio
    and the new Sharpe ratio.

    The ProfitHurdle algorithm lets the user calculate the required mean return for a strategy at a given level of
    significance, taking multiple testing into account.
    """

    def __init__(self):
        self.simulations = 2000

    @staticmethod
    def _sample_random_multest(rho, n_trails, prob_zero_mean, lambd, n_simulations, annual_vol=0.15, n_obs=240):
        """
        Generates empirical p-value distributions.

        The algorithm is described in the paper and is based on the model estimated by `Harvey, C.R., Y. Liu,
        and H. Zhu., … and the Cross-section of Expected Returns. Review of Financial Studies, forthcoming 2015`,
        referred to as HLZ.

        It provides a correlation adjustment when tests are correlated. Researchers propose a structural model to
        capture trading strategies’ underlying distribution. With probability p0 (prob_zero_mean), a strategy has
        a mean return of zero and therefore comes from the null distribution. With probability 1 – p0, a strategy
        has a nonzero mean and therefore comes from the alternative distribution - exponential.

        Parameters for this function are estimated in the _parameter_calculation function.

        :param rho: (float) Average correlation among returns
        :param n_trails: (int) Total number of trials inside a simulation
        :param prob_zero_mean: (float) Probability for a random factor to have a zero mean
        :param lambd: (float) Average of monthly mean returns for true strategies
        :param n_simulations: (int) Number of rows (simulations)
        :param annual_vol: (float) HLZ assume that the innovations in returns follow a normal distribution with a mean
                                   of zero and a standard deviation of ma = 15%
        :param n_obs: (int) Number of observations of used for volatility estimation from HLZ
        :return: (np.ndarray) array with distributions calculated
        """

        # Assumed level of monthly volatility = adjusted yearly volatility
        monthly_volatility = annual_vol / 12 ** (1 / 2)

        # Creating a correlation matrix of simulated returns. All correlations are assumed to be the same as average
        # correlation among returns
        # The first row of the correlation matrix: [1, rho, rho, .., rho]
        correlation_vector = np.insert(rho * np.ones((1, n_trails - 1)), 0, 1)
        # Correlation matrix created from the vector by expanding it
        correlation_matrix = linalg.toeplitz(correlation_vector)
        # Vector with mean of simulated returns - zeros
        mean = np.zeros(n_trails)

        # Creating a sample from a multivariate normal distribution as returns simulations
        # Covariance matrix - Created from correlation matrix multiplied by monthly volatility and adjusted
        covariance_matrix = correlation_matrix * (monthly_volatility ** 2 / n_obs)
        # Result - n_simulations rows with n_trails inside
        shock_mat = np.random.multivariate_normal(mean, covariance_matrix, n_simulations)

        # Sample of uniform distribution with same dimensions as shock_mat
        prob_vec = np.random.uniform(0, 1, (n_simulations, n_trails))
        # Sample of exponential distribution with same dimensions ad shock_mat
        mean_vec = np.random.exponential(lambd, (n_simulations, n_trails))

        # Taking the factors that have non-zero mean
        nonzero_mean = prob_vec > prob_zero_mean
        # Generating the null hypothesis - either zero mean or from an exponential distribution
        mu_null = np.multiply(nonzero_mean, mean_vec)
        # Matrix of p-value distributions
        tstat_matrix = abs(mu_null + shock_mat) / (monthly_volatility / n_obs ** (1 / 2))

        return tstat_matrix

    @staticmethod
    def _parameter_calculation(rho):
        """
        Estimates the parameters used to generate the distributions in _sample_random_multest

        Based on the work of HLZ, the pairwise correlation of returns is used to estimate the probability (prob_zero_mean),
        total number of trials (n_simulations) and (lambd) - parameter of the exponential distribution. Levels and
        parameters taken from the work of HLZ.

        :param rho: (float) average correlation coefficient between strategy returns
        :return: (np.array) array of parameters
        """
        # Levels of paramters based on rho. [rho, n_simulations, prob_zero_mean, lambd]
        parameter_levels = np.array([[0, 1295, 3.9660 * 0.1, 5.4995 * 0.001],
                                     [0.2, 1377, 4.4589 * 0.1, 5.5508 * 0.001],
                                     [0.4, 1476, 4.8604 * 0.1, 5.5413 * 0.001],
                                     [0.6, 1773, 5.9902 * 0.1, 5.5512 * 0.001],
                                     [0.8, 3109, 8.3901 * 0.1, 5.5956 * 0.001]])

        # Linear interpolation for parameter estimates
        if (rho >= 0) and (rho < 0.2):
            parameters = ((0.2 - rho) / 0.2) * parameter_levels[0] + ((rho - 0) / 0.2) * parameter_levels[1]
        elif (rho < 0.4):
            parameters = ((0.4 - rho) / 0.2) * parameter_levels[1] + ((rho - 0.2) / 0.2) * parameter_levels[2]
        elif (rho < 0.6):
            parameters = ((0.6 - rho) / 0.2) * parameter_levels[2] + ((rho - 0.4) / 0.2) * parameter_levels[3]
        elif (rho < 0.8):
            parameters = ((0.8 - rho) / 0.2) * parameter_levels[3] + ((rho - 0.6) / 0.2) * parameter_levels[4]
        elif (rho < 1.0): #Interpolation based on the previous level here
            parameters = ((0.8 - rho) / 0.2) * parameter_levels[3] + ((rho - 0.6) / 0.2) * parameter_levels[4]
        else:
            parameters = parameter_levels[1]  # Set at the preferred level if rho is misspecified

        return parameters

    @staticmethod
    def _annualized_sharpe_ratio(sharpe_ratio, sampling_frequency='A', rho=0, annualized=False, autocorr_adjusted=False):
        """
        Calculate the equivalent annualized Sharpe ratio after taking the autocorrelation of returns into account

        Adjustments are based on the work of `Lo, A., The Statistics of Sharpe Ratios. Financial Analysts Journal,
        58 (2002), pp. 36-52` and are described there in more details.

        :param sharpe_ratio: (float) Sharpe ratio of the strategy
        :param sampling_frequency: (str) Sampling frequency of returns
                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
        :param rho: (float) Autocorrelation coefficient of returns at specified frequency
        :param annualized: (bool) Flag if annualized, 'ind_an' = 1, otherwise = 0
        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
        :return: (float) Adjusted annualized Sharpe ratio
        """

        # If not annualized, calculating the appropriate multiplier for the Sharpe ratio
        if not annualized:
            if sampling_frequency == 'D':
                annual_multiplier = (360) ** (1 / 2)
            elif sampling_frequency == 'W':
                annual_multiplier = (52) ** (1 / 2)
            elif sampling_frequency == 'M':
                annual_multiplier = (12) ** (1 / 2)
            elif sampling_frequency == 'Q':
                annual_multiplier = (4) ** (1 / 2)
            elif sampling_frequency == 'A':
                annual_multiplier = 1
        else:
            annual_multiplier = 1

        # If not adjusted for returns autocorrelation, another multiplier
        if not autocorr_adjusted:
            if sampling_frequency == 'D':
                autocorr_multiplier = (1 + (2 * rho / (1 - rho)) * (1 - ((1 - rho ** (360)) / (360 * (1 - rho))))) ** (-0.5)
            elif sampling_frequency == 'W':
                autocorr_multiplier = (1 + (2 * rho / (1 - rho)) * (1 - ((1 - rho ** (52)) / (52 * (1 - rho))))) ** (-0.5)
            elif sampling_frequency == 'M':
                autocorr_multiplier = (1 + (2 * rho / (1 - rho)) * (1 - ((1 - rho ** (12)) / (12 * (1 - rho))))) ** (-0.5)
            elif sampling_frequency == 'Q':
                autocorr_multiplier = (1 + (2 * rho / (1 - rho)) * (1 - ((1 - rho ** (4)) / (4 * (1 - rho))))) ** (-0.5)
            elif sampling_frequency == 'A':
                autocorr_multiplier = 1
        else:
            autocorr_multiplier = 1

        # And calculating the adjusted Sharpe ratio
        adjusted_sr = sharpe_ratio * annual_multiplier * autocorr_multiplier

        return adjusted_sr

    @staticmethod
    def _monthly_observations(num_obs, sampling_frequency):
        """
        Calculates the number of monthly observations based on sampling frequency and number of observations

        :param num_obs: (int) number of observations used for modelling
        :param:sampling_frequency: (str) Sampling frequency of returns
                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
        :return: (np.float64) number of monthly observations
        """
        # N - Number of monthly observations
        if sampling_frequency == 'D':
            monthly_obs = np.floor(num_obs * 12 / 360)
        elif sampling_frequency == 'W':
            monthly_obs = np.floor(num_obs * 12 / 52)
        elif sampling_frequency == 'M':
            monthly_obs = np.floor(num_obs * 12 / 12)
        elif sampling_frequency == 'Q':
            monthly_obs = np.floor(num_obs * 12 / 4)
        elif sampling_frequency == 'A':
            monthly_obs = np.floor(num_obs * 12 / 1)
        else:  # If the frequency is misspecified
            monthly_obs = np.floor(num_obs)

        return monthly_obs

    def haircut_sharpe_ratios(self, sampling_frequency, num_obs, sharpe_ratio, annualized,
                              autocorr_adjusted, rho_a, num_mult_test, rho):
        # pylint: disable=invalid-name, too-many-branches
        """
        Calculate Sharpe ratio adjustments due to testing multiplicity

        This algorithm lets the user calculate Sharpe ratio adjustments and the corresponding haircuts based on
        key parameters of data used in the strategy backtesting. For each of the adjustment methods - Bonferroni,
        Holm, BHY (Benjamini, Hochberg and Yekutieli) and the Average the algorithm calculates adjusted p-value,
        haircut Sharpe ratio and the percentage of a haircut.

        The haircut is the percentage difference between the original Sharpe ratio and the new Sharpe ratio.

        :param sampling_frequency: (str) Sampling frequency ['D','W','M','Q','A'] of returns
        :param num_obs: (int) Number of returns in the frequency specified in the previous step
        :param sharpe_ratio: (float) Sharpe ratio of the strategy. Either annualized or in the frequency specified in the previous step
        :param annualized: (bool) Flag if Sharpe ratio is annualized
        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
        :param rho_a: (float) Autocorrelation coefficient of returns at the specified frequency (if the Sharpe ratio
                              wasn't corrected)
        :param num_mult_test: (int) Number of tests in multiple testing allowed (HLZ (2015) find at least 315 factors)
        :param rho: (float) Average correlation among strategy returns
        :return: (np.ndarray) array with adjuted p-value, haircut sharpe ratio, percentage haircut as elements in a row
                              for Bonferroni, Holm, BHY and average adjustment as rows
        """
        # Calculating the annual Sharpe ratio adjusted for the autocorrelation of returns
        sr_annual = self._annualized_sharpe_ratio(sharpe_ratio, sampling_frequency, rho_a, annualized, autocorr_adjusted)

        # Estimating the parameters used for distributions based on HLZ model
        # Result is [rho, n_simulations, prob_zero_mean, lambd]
        parameters = self._parameter_calculation(rho)

        # Getting the number of monthly observations in a sample
        monthly_obs = self._monthly_observations(num_obs, sampling_frequency)

        # Needed number of trails inside a simulation with the check of (num_simulations >= num_mul_tests)
        num_trails = int((np.floor(num_mult_test / parameters[1]) + 1) * np.floor(parameters[1] + 1))
        # Generating a panel of t-ratios (of size self.simulations * num_simulations)
        t_sample = self._sample_random_multest(parameters[0], num_trails, parameters[2], parameters[3], self.simulations)

        # Constant used in BHY method
        index_vector = np.arange(1, num_mult_test + 1)
        c_constant = sum(1 / index_vector)

        # Annual Sharpe ratio, adjusted to monthly
        sr_monthly = sr_annual / 12 ** (1 / 2)
        # Calculating t-ratio based on the Sharpe ratio and the number of observations
        t_ratio = sr_monthly * monthly_obs ** (1 / 2)
        # Calculating adjusted p-value from the given t-ratio
        p_val = 2 * (1 - ss.t.cdf(t_ratio, monthly_obs - 1))

        # Creating arrays for p-values from simulations of Holm and BHY methods.
        p_holm = np.ones(self.simulations)
        p_bhy = np.ones(self.simulations)

        # Iterating through the simulations
        for simulation_number in range(1, self.simulations + 1):
            # Test print to measure speed
            if simulation_number % 100 == 0:
                print(simulation_number)

            # Get one sample of previously generated simulation of t-values
            t_values_simulation = t_sample[simulation_number - 1, 1:(num_mult_test + 1)]
            # Calculating adjusted p-values from the simulated t-ratios
            p_values_simulation = 2 * (1 - ss.norm.cdf(t_values_simulation, 0, 1))

            # Holm method

            # To the N (num_mult_test) other strategies tried (from the simulation),
            # we add the adjusted p_value of the real strategy.
            all_p_values = np.append(p_values_simulation, p_val)
            # Ordering p-values
            all_p_values = np.sort(all_p_values)

            # Array for final p-values of the Holm method
            p_holm_values = np.array([])
            # Iterating through multiple tests
            for i in range(1, (num_mult_test + 2)):
                # Creating array for Holm adjusted p-values (M-j+1)*p(j) in the paper
                p_adjusted_holm = np.array([])
                # Iterating through the available subsets of Holm adjusted p-values
                for j in range(1, i + 1):
                    # Holm adjusted p-values
                    p_adjusted_holm = np.append(p_adjusted_holm, (num_mult_test + 1 - j + 1) * all_p_values[j - 1])
                # Calculating the final p-values of the Holm method and adding to array
                p_holm_values = np.append(p_holm_values, min(max(p_adjusted_holm), 1))

            # Getting the Holm adjusted p-value that is significant at our p_val level
            p_holm_significant = p_holm_values[all_p_values == p_val]
            # Adding this value to our array of simulations
            p_holm[simulation_number - 1] = p_holm_significant[0]

            # BHY method

            # Array for final p-values of the BHY method
            p_bhy_values = np.array([])

            # Iterating through multiple tests
            for i in range(1, num_mult_test + 2):
                # Iterating backwards here
                kk = (num_mult_test + 1) - (i - 1)
                if kk == (num_mult_test + 1):  # If it's the last observation
                    # The p-value stays the same
                    p_adjusted_holm = all_p_values[-1]
                else: # If it's the previous observations
                    # The p-value is adjusted according to the BHY method
                    p_adjusted_holm = min(((num_mult_test + 1) * c_constant / kk) * all_p_values[kk - 1], p_previous)
                # Adding the final BHY method p-values to array
                p_bhy_values = np.append(p_adjusted_holm, p_bhy_values)
                p_previous = p_adjusted_holm

            # Getting the BHY adjusted p-value that is significant at our p_val level
            p_holm_significant = p_bhy_values[all_p_values == p_val]
            # Adding this value to our array of simulations
            p_bhy[simulation_number - 1] = p_holm_significant[0]

        # Calculating the resulting p-values of methods from simulations
        # Bonferroni
        p_BON = np.minimum(num_mult_test * p_val, 1)
        # Holm
        p_HOL = np.median(p_holm)
        # BHY
        p_BHY = np.median(p_bhy)
        # Average
        p_avg = (p_BON + p_HOL + p_BHY) / 3

        # Inverting to get z-score for every method
        z_BON = ss.t.ppf(1 - p_BON / 2, monthly_obs - 1)
        z_HOL = ss.t.ppf(1 - p_HOL / 2, monthly_obs - 1)
        z_BHY = ss.t.ppf(1 - p_BHY / 2, monthly_obs - 1)
        z_avg = ss.t.ppf(1 - p_avg / 2, monthly_obs - 1)

        # Adjusted annualized Sharpe ratio of methods
        sr_BON = (z_BON / monthly_obs ** (1 / 2)) * 12 ** (1 / 2)
        sr_HOL = (z_HOL / monthly_obs ** (1 / 2)) * 12 ** (1 / 2)
        sr_BHY = (z_BHY / monthly_obs ** (1 / 2)) * 12 ** (1 / 2)
        sr_avg = (z_avg / monthly_obs ** (1 / 2)) * 12 ** (1 / 2)

        # Haircut of the Sharpe ratio for every method
        hc_BON = (sr_annual - sr_BON) / sr_annual
        hc_HOL = (sr_annual - sr_HOL) / sr_annual
        hc_BHY = (sr_annual - sr_BHY) / sr_annual
        hc_avg = (sr_annual - sr_avg) / sr_annual

        results = np.array([[p_BON, sr_BON, hc_BON * 100],
                            [p_HOL, sr_HOL, hc_HOL * 100],
                            [p_BHY, sr_BHY, hc_BHY * 100],
                            [p_avg, sr_avg, hc_avg * 100]])

        return results
