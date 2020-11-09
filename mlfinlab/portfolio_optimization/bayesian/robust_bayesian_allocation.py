#pylint: disable=missing-docstring
import pandas as pd
import numpy as np
from scipy.stats.distributions import chi2
from mlfinlab.portfolio_optimization.modern_portfolio_theory import MeanVarianceOptimisation


class RobustBayesianAllocation:
    """
    This class implements the Robust Bayesian Allocation (RBA) algorithm from the following paper: Meucci, A., 2011. Robust Bayesian Allocation.
    Instead of relying on historical sample data, this method combines information from the sample distribution and investor-specified prior distribution
    to calculate the posterior market distribution. Finally, the algorithm generates a Bayesian efficient frontier using this posterior and selects the
    robust bayesian portfolio from it.
    """

    def __init__(self, discretisations=10):
        """
        Initialise.

        Class Variables:

        - ``discretisations`` - (int) Number of portfolios to generate along the bayesian efficient frontier. The final robust portfolio
                                      will be chosen from the set of these portfolios.
        - ``weights`` - (pd.DataFrame) Final portfolio weights.
        - ``portfolio_return`` - (float) Portfolio return.
        - ``portfolio_risk`` - (float) Portfolio variance/risk.
        - ``posterior_mean`` - (pd.DataFrame) Posterior mean returns for assets in portfolio.
        - ``posterior_covariance`` - (pd.DataFrame) Posterior covariance matrix of asset returns.
        - ``posterior_mean_confidence`` - (float) Investor confidence in the posterior mean distribution.
        - ``posterior_covariance_confidence`` - (float) Investor confidence in the posterior covariance distribution.
        """

        pass

    def allocate(self, prior_mean, prior_covariance, sample_mean, sample_covariance, sample_size, relative_confidence_in_prior_mean=1,
                 relative_confidence_in_prior_covariance=1, posterior_mean_estimation_risk_level=0.1, posterior_covariance_estimation_risk_level=0.1,
                 max_volatility=1.0, asset_names=None):
        #pylint: disable=too-many-arguments, invalid-name
        """
        Combines the prior and sample distributions to calculate a robust bayesian portfolio.

        :param prior_mean: (Numpy array/Python list) The mean returns of the prior distribution.
        :param prior_covariance: (pd.DataFrame/Numpy matrix) The covariance of the prior distribution.
        :param sample_mean: (Numpy array/Python list) The mean returns of sample distribution.
        :param sample_covariance: (pd.DataFrame/Numpy matrix) The covariance of returns of the sample distribution.
        :param sample_size: (int) Number of observations in the data used to estimate sample means and covariance.
        :param relative_confidence_in_prior_mean: (float) A numeric value specifying the investor's confidence in the mean of the prior distribution.
                                                          This confidence is measured relative to the sample distribution.
        :param relative_confidence_in_prior_covariance: (float) A numeric value specifying the investor's confidence in the covariance of the prior
                                                                distribution. This confidence is measured relative to the sample distribution.
        :param posterior_mean_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior mean. Lower value corresponds
                                                             to less confidence and a more aggressive investor while a higher value will result in a more
                                                             conservative portfolio.
        :param posterior_covariance_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior covariance. Lower value
                                                                   corresponds to less confidence and a more aggressive investor while a higher value will
                                                                   result in a more conservative portfolio.
        :param max_volatility: (float) The maximum preferred volatility of the final robust portfolio.
        :param asset_names: (Numpy array/Python list) List of asset names in the portfolio.
        """

        pass

    @staticmethod
    def _pre_process_inputs(prior_mean, prior_covariance, sample_mean, sample_covariance):
        """
        Initial preprocessing of inputs.

        :param prior_mean: (Numpy array/Python list) The mean returns of the prior distribution.
        :param prior_covariance: (pd.DataFrame/Numpy matrix) The covariance of the prior distribution.
        :param sample_mean: (Numpy array/Python list) The mean returns of sample distribution.
        :param sample_covariance: (pd.DataFrame/Numpy matrix) The covariance of returns of the sample distribution.
        :return: (Numpy array, Numpy matrix, Numpy array, Numpy matrix) Same inputs but converted to numpy arrays and matrices.
        """

        pass

    def _find_robust_portfolio(self, bayesian_portfolios, bayesian_portfolio_volatilities, bayesian_portfolio_returns, gamma_mean,
                               gamma_covariance, asset_names):
        """
        From the set of portfolios along the bayesian efficient frontier, select the robust portfolio - one which gives highest return for highest risk.

        :param bayesian_portfolios: (Python list) List of portfolio weights along the bayesian efficient frontier
        :param bayesian_portfolio_volatilities: (Python list) Volatilities of portfolios along the bayesian efficient frontier
        :param bayesian_portfolio_returns: (Python list) Expected returns of portfolios along the bayesian efficient frontier
        :param gamma_mean: (float) Gamma value for the mean.
        :param gamma_covariance: (float) Gamma value for the covariance.
        :param asset_names: (Numpy array/Python list) List of asset names.
        """

        pass

    def _calculate_gamma(self, posterior_mean_estimation_risk_level, posterior_covariance_estimation_risk_level, max_volatility, num_assets):
        # pylint: disable=invalid-name
        """
        Calculate the gamma values appearing in the robust bayesian allocation objective and risk constraint.

        :param posterior_mean_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior mean. Lower value corresponds
                                                             to less confidence and a more conservative investor while a higher value will result in a more
                                                             risky portfolio.
        :param posterior_covariance_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior covariance. Lower value
                                                                   corresponds to less confidence and a more conservative investor while a higher value will
                                                                   result in a more risky portfolio.
        :param max_volatility: (float) The maximum preferred volatility of the final robust portfolio.
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (float, float) gamma mean, gamma covariance
        """

        pass

    def _calculate_posterior_distribution(self, sample_size, relative_confidence_in_prior_mean, relative_confidence_in_prior_covariance, sample_mean,
                                          sample_covariance, prior_mean, prior_covariance):
        """
        Calculate the posterior market distribution from prior and sample distributions.

        :param sample_size: (int) Number of observations in the data used to estimate sample means and covariance.
        :param relative_confidence_in_prior_mean: (float) A numeric value specifying the investor's confidence in the mean of the prior distribution.
                                                          This confidence is measured relative to the sample distribution.
        :param relative_confidence_in_prior_covariance: (float) A numeric value specifying the investor's confidence in the covariance of the prior
                                                                distribution. This confidence is measured relative to the sample distribution.
        :param sample_mean: (Numpy array) The mean returns of sample distribution.
        :param sample_covariance: (Numpy matrix) The covariance of returns of the sample distribution.
        :param prior_mean: (Numpy array) The mean returns of the prior distribution.
        :param prior_covariance: (Numpy matrix) The covariance of the prior distribution.
        """

        pass

    def _calculate_bayesian_frontier(self, asset_names):
        """
        Generate portfolios along the bayesian efficient frontier.

        :param asset_names: (Numpy array/Python list) List of asset names in the portfolio.
        :return: (Python list, Python list, Python list) Portfolios along the bayesian efficient frontier.
        """

        pass

    @staticmethod
    def _error_checks(prior_mean, prior_covariance, sample_mean, sample_covariance, sample_size, relative_confidence_in_prior_mean,
                      relative_confidence_in_prior_covariance, posterior_mean_estimation_risk_level, posterior_covariance_estimation_risk_level,
                      max_volatility):
        #pylint: disable=invalid-name
        """
        Initial error checks on inputs.

        :param prior_mean: (Numpy array/Python list) The mean returns of the prior distribution.
        :param prior_covariance: (pd.DataFrame/Numpy matrix) The covariance of the prior distribution.
        :param sample_mean: (Numpy array/Python list) The mean returns of sample distribution.
        :param sample_covariance: (pd.DataFrame/Numpy matrix) The covariance of returns of the sample distribution.
        :param sample_size: (int) Number of observations in the data used to estimate sample means and covariance.
        :param relative_confidence_in_prior_mean: (float) A numeric value specifying the investor's confidence in the mean of the prior distribution.
                                                          This confidence is measured relative to the sample distribution.
        :param relative_confidence_in_prior_covariance: (float) A numeric value specifying the investor's confidence in the covariance of the prior
                                                                distribution. This confidence is measured relative to the sample distribution.
        :param posterior_mean_estimation_risk_level: (float) Denotes the confidence of investor to estimation risk in posterior mean. Lower value corresponds
                                                             to less confidence and a more conservative investor while a higher value will result in a more
                                                             risky portfolio.
        :param posterior_covariance_estimation_risk_level: (float) Denotes the aversion of investor to estimation risk in posterior covariance. Lower value
                                                                   corresponds to less confidence and a more conservative investor while a higher value will
                                                                   result in a more risky portfolio.
        :param max_volatility: (float) The maximum preferred volatility of the final robust portfolio.
        """

        pass
