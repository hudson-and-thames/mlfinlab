# pylint: disable=missing-module-docstring
import numbers
from math import log, ceil
import numpy as np
import pandas as pd

from mlfinlab.portfolio_optimization.estimators import ReturnsEstimators
from mlfinlab.portfolio_optimization.estimators import RiskEstimators


class CriticalLineAlgorithm:
    # pylint: disable=too-many-instance-attributes
    """
    This class implements the famous Critical Line Algorithm (CLA) for mean-variance portfolio optimisation. It is reproduced with
    modification from the following paper: `D.H. Bailey and M.L. Prado “An Open-Source Implementation of the Critical- Line
    Algorithm for Portfolio Optimization”,Algorithms, 6 (2013), 169-196. <http://dx.doi.org/10.3390/a6010169>`_.

    The Critical Line Algorithm is a famous portfolio optimisation algorithm used for calculating the optimal allocation weights
    for a given portfolio. It solves the optimisation problem with optimisation constraints on each weight - lower and upper
    bounds on the weight value. This class can compute multiple types of solutions:

    1. CLA Turning Points
    2. Minimum Variance
    3. Maximum Sharpe
    4. Efficient Frontier Allocations
    """

    def __init__(self, weight_bounds=(0, 1), calculate_expected_returns="mean"):
        """
        Initialise the storage arrays and some preprocessing.

        :param weight_bounds: (tuple) A tuple specifying the lower and upper bound ranges for the portfolio weights.
        :param calculate_expected_returns: (str) The method to use for calculation of expected returns.
                                                 Currently supports ``mean`` and ``exponential``
        """

        pass

    def allocate(self, asset_names=None, asset_prices=None, expected_asset_returns=None, covariance_matrix=None,
                 solution="cla_turning_points", resample_by=None):
        # pylint: disable=consider-using-enumerate,too-many-locals,too-many-branches,too-many-statements,bad-continuation
        """
        Calculate the portfolio asset allocations using the method specified.

        :param asset_names: (list) List of strings containing the asset names.
        :param asset_prices: (pd.Dataframe) Dataframe of historical asset prices (adj closed).
        :param expected_asset_returns: (list) List of mean asset returns (mu).
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns.
        :param solution: (str) Specifies the type of solution to compute. Supported strings: ``cla_turning_points``, ``max_sharpe``,
                               ``min_volatility``, ``efficient_frontier``.
        :param resample_by: (str) Specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling.
        """

        pass

    def _initialise(self, asset_prices, expected_asset_returns, covariance_matrix, resample_by):
        # pylint: disable=invalid-name, too-many-branches, bad-continuation
        """
        Initialise covariances, upper-bounds, lower-bounds and storage buffers.
        :param asset_prices: (pd.Dataframe) Dataframe of asset prices indexed by date.
        :param expected_asset_returns: (list) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.Dataframe) User supplied dataframe of asset returns indexed by date. Used for
                                                 calculation of covariance matrix.
        :param resample_by: (str) Specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling.
        :return: (Numpy matrix, Numpy array, Numpy array, Numpy array) Data matrices and bounds arrays.
        """

        pass

    @staticmethod
    def _infnone(number):
        """
        Converts a Nonetype object to inf.

        :param number: (int/float/None) Number.
        :return: (float) -inf or number.
        """

        pass

    @staticmethod
    def _init_algo(expected_returns, lower_bounds, upper_bounds):
        """
        Initial setting up of the algorithm. Calculates the first free weight of the first turning point.

        :param expected_asset_returns: (Numpy array) A list of mean stock returns (mu).
        :param lower_bounds: (Numpy array) An array containing lower bound values for the weights.
        :param upper_bounds: (Numpy array) An array containing upper bound values for the weights.
        :return: (list, list) asset index and the corresponding free weight value.
        """

        pass

    @staticmethod
    def _compute_bi(c_final, asset_bounds_i):
        """
        Calculates which bound value to assign to a bounded asset - lower bound or upper bound.

        :param c_final: (float) A value calculated using the covariance matrices of free weights.
                                Refer to https://pdfs.semanticscholar.org/4fb1/2c1129ba5389bafe47b03e595d098d0252b9.pdf
                                for more information.
        :param asset_bounds_i: (list) A list containing the lower and upper bound values for the ith weight.
        :return: (float) Bounded weight value.
        """

        pass

    def _compute_w(self, covar_f_inv, covar_fb, mean_f, w_b):
        """
        Compute the turning point associated with the current set of free weights F.

        :param covar_f_inv: (np.array) Inverse of covariance matrix of free assets.
        :param covar_fb: (np.array) Covariance matrix between free assets and bounded assets.
        :param mean_f: (np.array) Expected returns of free assets.
        :param w_b: (np.array) Bounded asset weight values.
        :return: (array, float) List of turning point weights and gamma value from the lagrange equation.
        """

        pass

    def _compute_lambda(self, covar_f_inv, covar_fb, mean_f, w_b, asset_index, b_i):
        """
        Calculate the lambda value in the lagrange optimsation equation.

        :param covar_f_inv: (np.array) Inverse of covariance matrix of free assets.
        :param covar_fb: (np.array) Covariance matrix between free assets and bounded assets.
        :param mean_f: (np.array) Expected returns of free assets.
        :param w_b: (np.array) Bounded asset weight values.
        :param asset_index: (int) Index of the asset in the portfolio.
        :param b_i: (list) List of upper and lower bounded weight values.
        :return: (float) Lambda value.
        """

        pass

    def _get_matrices(self, cov_matrix, expected_returns,free_weights):
        """
        Calculate the required matrices between free and bounded assets.

        :param cov_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        :param free_weights: (list) List of free assets/weights.
        :return: (tuple of np.array matrices) The corresponding matrices.
        """

        pass

    def _get_bounded_weights(self, expected_returns, free_weights):
        """
        Compute the list of bounded assets.

        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        :param free_weights: (np.array) List of free weights/assets.
        :return: (np.array) List of bounded assets/weights.
        """


        pass

    @staticmethod
    def _diff_lists(list_1, list_2):
        """
        Calculate the set difference between two lists.

        :param list_1: (list) A list of asset indices.
        :param list_2: (list) Another list of asset indices.
        :return: (list) Set difference between the two input lists.
        """

        pass

    @staticmethod
    def _reduce_matrix(matrix, row_indices, col_indices):
        """
        Reduce a matrix to the provided set of rows and columns.

        :param matrix: (np.array) A matrix whose subset of rows and columns we need.
        :param row_indices: (list) List of row indices for the matrix.
        :param col_indices: (list) List of column indices for the matrix.
        :return: (np.array) Subset of input matrix.
        """


        pass

    def _purge_num_err(self, lower_bounds, upper_bounds, tol):
        """
        Purge violations of inequality constraints (associated with ill-conditioned cov matrix).

        :param lower_bounds: (Numpy array) An array containing lower bound values for the weights.
        :param upper_bounds: (Numpy array) An array containing upper bound values for the weights.
        :param tol: (float) Tolerance level for purging.
        """

        pass

    def _purge_excess(self, expected_returns):
        """
        Remove violations of the convex hull.

        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        """

        pass

    @staticmethod
    def _golden_section(obj, left, right, **kwargs):
        """
        Golden section method. Maximum if kargs['minimum']==False is passed.

        :param obj: (function) The objective function on which the extreme will be found.
        :param left: (float) The leftmost extreme of search.
        :param right: (float) The rightmost extreme of search.
        """

        pass

    @staticmethod
    def _eval_sr(alpha, cov_matrix, expected_returns, w_0, w_1):
        """
        Evaluate the sharpe ratio of the portfolio within the convex combination.

        :param alpha: (float) Convex combination value.
        :param cov_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        :param w_0: (list) First endpoint of convex combination of weights.
        :param w_1: (list) Second endpoint of convex combination of weights.
        :return:
        """

        pass

    def _bound_free_weight(self, cov_matrix, expected_returns, lower_bounds, upper_bounds, free_weights):
        """
        Add a free weight to list of bounded weights.

        :param cov_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        :param lower_bounds: (Numpy array) An array containing lower bound values for the weights.
        :param upper_bounds: (Numpy array) An array containing upper bound values for the weights.
        :param free_weights: (list) List of free-weight indices.
        :return: (float, int, int) Lambda value, index of free weight to be bounded, bound weight value.
        """

        pass

    def _free_bound_weight(self, cov_matrix, expected_returns, free_weights):
        """
        Add a bounded weight to list of free weights.

        :param cov_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        :param free_weights: (list) List of free-weight indices.
        :return: (float, int) Lambda value, index of the bounded weight to be made free.
        """

        pass

    def _compute_solution(self, assets, solution, covariance_matrix, expected_returns):
        """
        Compute the desired solution to the portfolio optimisation problem.

        :param assets: (list) A list of asset names.
        :param solution: (str) Specify the type of solution to compute. Options are: cla_turning_points, max_sharpe,
                               min_volatility, efficient_frontier.
        :param covariance_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        """

        pass

    def _max_sharpe(self, cov_matrix, expected_returns):
        """
        Compute the maximum sharpe portfolio allocation.

        :param cov_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        :return: (float, np.array) Tuple of max. sharpe value and the set of weight allocations.
        """

        pass

    def _min_volatility(self, cov_matrix):
        """
        Compute minimum volatility portfolio allocation.

        :param cov_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :return: (float, np.array) Tuple of minimum variance value and the set of weight allocations.
        """

        pass

    def _efficient_frontier(self, cov_matrix, expected_returns, points=100):
        # pylint: disable=invalid-name
        """
        Compute the entire efficient frontier solution.

        :param cov_matrix: (Numpy matrix) Covariance matrix of asset returns.
        :param expected_returns: (Numpy array) Array of mean asset returns (mu).
        :param points: (int) Number of efficient frontier points to be calculated.
        :return: (tuple) Tuple of mean, variance amd weights of the frontier solutions.
        """

        pass
