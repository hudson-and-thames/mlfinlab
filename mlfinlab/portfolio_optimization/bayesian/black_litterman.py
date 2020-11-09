#pylint: disable=missing-docstring
import pandas as pd
import numpy as np
from numpy.linalg import inv


class VanillaBlackLitterman:
    """
    This class contains the original (vanilla) implementation of the Black-Litterman (BL) portfolio allocation algorithm
    first proposed in `Black, F. and Litterman, R., 1992. Global portfolio optimization. Financial analysts journal, 48(5),
    pp.28-43 <https://faculty.fuqua.duke.edu/~charvey/Teaching/BA453_2006/blacklitterman.pdf>`_.

    Instead of relying on empirical estimates of expected returns, BL starts with market equilibrium returns and combines
    them with external views to generate expected returns which are in sync  with the investor's specific views on the market.
    One of the main components of the model is the omega matrix which represents the error in the specified views. This class
    currently includes two commonly used methods for determining this matrix:

        1. Prior Variance: Uses the empirical covariance (prior) for forming the omega matrix.
        2. User Specified Confidences: Combines user-specified confidences on the views to form the omega matrix.
    """

    def __init__(self):
        """
        Initialise.

        Class Variables:

        - ``weights`` - (pd.DataFrame) Final portfolio weights.
        - ``implied_equilibrium_returns`` - (pd.DataFrame) CAPM implied mean equilibrium returns.
        - ``posterior_expected_returns`` - (pd.DataFrame) Posterior BL expected returns.
        - ``posterior_covariance`` - (pd.DataFrame) Posterior BL covariance matrix.
        """

        pass

    def allocate(self, covariance, market_capitalised_weights, investor_views, pick_list, omega=None, risk_aversion=2.5, tau=0.05,
                 omega_method='prior_variance', view_confidences=None, asset_names=None):
        # pylint: disable=too-many-arguments
        """
        Calculate asset allocations using Black-Litterman algorithm.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param market_capitalised_weights: (Numpy array/Python list) List of market capitalised weights of assets used for calculating the
                                                                     implied excess equilibrium returns.
        :param investor_views: (Numpy array/Python list) User-specified list of views expressed in the form of percentage excess returns.
        :param pick_list: (Numpy array/Python list) List of dictionaries specifying which assets involved in the respective view.
                                                    Let's say you have 3 assets 'A', 'B' and 'C' You want to specify the following
                                                    views: "A will yield an expected return of 5%" and "B will outperform C by 6%".
                                                    The pick list for the above views will be something like this: [{ 'A': 1 }, { 'B': 1, 'C': -1 }]
        :param omega: (pd.DataFrame/Numpy matrix) Diagonal matrix of variance in investor views.
        :param risk_aversion: (float) Quantifies the risk-averse nature of the investor - a higher value means more risk averse and vice-versa.
        :param tau: (float) Constant of proportionality. Typical values range between 0 and 1.
        :param omega_method: (str) The type of method to use for calculating the omega matrix. Supported strings - ``prior_variance``, ``user_confidences``.
        :param view_confidences: (Numpy array/Python list) Use supplied confidences for the views. The confidences are specified
                                                           in percentages e.g. 0.05, 0.4, 0.9 etc.... This parameter is required when the
                                                           omega method is set to ``user_confidences``.
        :param asset_names: (Numpy array/Python list) A list of strings specifying the asset names.
        """

        pass

    @staticmethod
    def _pre_process_inputs(covariance, market_capitalised_weights, investor_views):
        """
        Initial preprocessing of inputs.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param market_capitalised_weights: (Numpy array/Python list) List of market capitalised weights of assets.
        :param investor_views: (Numpy array/Python list) User-specified list of views expressed in the form of percentage excess returns.
        :return: (Numpy matrix, Numpy array, Numpy matrix) Preprocessed inputs.
        """

        pass

    def _calculate_max_sharpe_weights(self):
        """
        Calculate the Maximum Sharpe Ratio portfolio.

        :return: (Numpy array) Final portfolio weights.
        """

        pass

    def _calculate_posterior_expected_returns(self, covariance, tau, pick_matrix, omega, investor_views):
        """
        Calculate Black-Litterman expected returns from investor views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param tau: (float) Constant of proportionality.
        :param pick_matrix: (Numpy matrix) Matrix specifying which assets involved in the respective view.
        :param omega: (Numpy matrix) Diagonal matrix of variance in investor views.
        :param investor_views: (Numpy array/Python list) User-specified list of views expressed in the form of percentage excess returns.
        :return: (Numpy array) Posterior expected returns.
        """

        pass

    @staticmethod
    def _calculate_posterior_covariance(covariance, tau, pick_matrix, omega):
        """
        Calculate Black-Litterman covariance of asset returns from investor views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param tau: (float) Constant of proportionality
        :param pick_matrix: (Numpy matrix) Matrix specifying specifying which assets involved in the respective view.
        :param omega: (Numpy matrix) Diagonal matrix of variance in investor views.
        :return: (Numpy array) Posterior covariance of asset returns.
        """

        pass

    @staticmethod
    def _calculate_implied_equilibrium_returns(risk_aversion, covariance, market_capitalised_weights):
        """
        Calculate the CAPM implied equilibrium market weights using the reverse optimisation trick.

        :param risk_aversion: (float) Quantifies the risk averse nature of the investor - a higher value means more risk averse and vice-versa.
        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param market_capitalised_weights: (Numpy array/Python list) List of market capitalised weights of portfolio assets.
        :return: (Numpy array) Market equilibrium weights.
        """

        pass

    @staticmethod
    def _create_pick_matrix(num_views, num_assets, pick_list, asset_names):
        """
        Calculate the picking matrix that specifies which assets are involved in the accompanying views.

        :param num_views: (int) Number of views.
        :param num_assets: (int) Number of assets in the portfolio.
        :param pick_list: (Numpy array/Python list) List of dictionaries specifying which assets involved in the respective view.
        :param asset_names: (Numpy array/Python list) A list of strings specifying the asset names.
        :return: (Numpy matrix) Picking matrix.
        """

        pass

    def _calculate_omega(self, covariance, tau, pick_matrix, view_confidences, omega_method):
        """
        Calculate the omega matrix - uncertainty in investor views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param tau: (float) Constant of proportionality
        :param pick_matrix: (Numpy matrix) Matrix specifying specifying which assets involved in the respective view.
        :param view_confidences: (Numpy array/Python list) Use supplied confidences for the views. The confidences are specified
                                                           in percentages e.g. 0.05, 0.4, 0.9 etc....
        :param omega_method: (str) The type of method to use for calculating the omega matrix.
        :return: (Numpy matrix) Omega matrix.
        """

        pass

    @staticmethod
    def _calculate_idzorek_omega(covariance, view_confidences, pick_matrix):
        """
        Calculate the Idzorek omega matrix by taking into account user-supplied confidences in the views.

        :param covariance: (pd.DataFrame/Numpy matrix) The covariance matrix of asset returns.
        :param view_confidences: (Numpy array/Python list) Use supplied confidences for the views. The confidences are specified
                                                           in percentages e.g. 0.05, 0.4, 0.9 etc....
        :param pick_matrix: (Numpy matrix) Matrix specifying specifying which assets involved in the respective view.
        :return: (Numpy matrix) Idzorek Omega matrix.
        """

        pass

    def _post_processing(self, asset_names):
        """
        Final post processing of weights, expected returns and covariance matrix.

        :param asset_names: (Numpy array/Python list) A list of strings specifying the asset names.
        """

        pass

    @staticmethod
    def _error_checks(investor_views, pick_list, omega_method, view_confidences):
        """
        Perform initial warning checks.

        :param investor_views: (Numpy array/Python list) User-specified list of views expressed in the form of percentage excess returns.
        :param pick_list: (Numpy array/Python list) List of dictionaries specifying which assets involved in the respective view.
        :param omega_method: (str) The type of method to use for calculating the omega matrix.
        :param view_confidences: (Numpy array/Python list) Use supplied confidences for the views. The confidences are specified
                                                           in percentages e.g. 0.05, 0.4, 0.9 etc....
        """

        pass
