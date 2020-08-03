# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd


class RiskMetrics:
    """
    This class contains methods for calculating common risk metrics used in trading and asset management.
    """

    def __init__(self):

        pass

    @staticmethod
    def calculate_variance(covariance, weights):
        """
        Calculate the variance of a portfolio.

        :param covariance: (pd.DataFrame/np.matrix) Covariance matrix of assets
        :param weights: (list) List of asset weights
        :return: (float) Variance of a portfolio
        """


        pass

    @staticmethod
    def calculate_value_at_risk(returns, confidence_level=0.05):
        """
        Calculate the value at risk (VaR) of a portfolio/asset.

        :param returns: (pd.DataFrame/np.array) Historical returns for an asset / portfolio
        :param confidence_level: (float) Confidence level (alpha)
        :return: (float) VaR
        """

        pass

    def calculate_expected_shortfall(self, returns, confidence_level=0.05):
        """
        Calculate the expected shortfall (CVaR) of a portfolio/asset.

        :param returns: (pd.DataFrame/np.array) Historical returns for an asset / portfolio
        :param confidence_level: (float) Confidence level (alpha)
        :return: (float) Expected shortfall
        """

        pass

    @staticmethod
    def calculate_conditional_drawdown_risk(returns, confidence_level=0.05):
        """
        Calculate the conditional drawdown of risk (CDaR) of a portfolio/asset.

        :param returns: (pd.DataFrame/np.array) Historical returns for an asset / portfolio
        :param confidence_level: (float) Confidence level (alpha)
        :return: (float) Conditional drawdown risk
        """

        pass
