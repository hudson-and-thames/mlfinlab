# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram
from scipy.spatial.distance import squareform
from mlfinlab.portfolio_optimization.estimators import ReturnsEstimators, RiskEstimators
from mlfinlab.portfolio_optimization.utils import RiskMetrics


class HierarchicalRiskParity:
    """
    This class implements the Hierarchical Risk Parity algorithm mentioned in the following paper: `López de Prado, Marcos,
    Building Diversified Portfolios that Outperform Out-of-Sample (May 23, 2016). Journal of Portfolio Management,
    2016 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_; The code is reproduced with modification from his book:
    Advances in Financial Machine Learning, Chp-16
    By removing exact analytical approach to the calculation of weights and instead relying on an approximate
    machine learning based approach (hierarchical tree-clustering), Hierarchical Risk Parity produces weights which are stable to
    random shocks in the stock-market. Moreover, previous algorithms like CLA involve the inversion of covariance matrix which is
    a highly unstable operation and tends to have major impacts on the performance due to slight changes in the covariance matrix.
    By removing dependence on the inversion of covariance matrix completely, the Hierarchical Risk Parity algorithm is fast,
    robust and flexible.
    """

    def __init__(self):

        pass

    def allocate(self,
                 asset_names=None,
                 asset_prices=None,
                 asset_returns=None,
                 covariance_matrix=None,
                 distance_matrix=None,
                 side_weights=None,
                 linkage='single'):
        # pylint: disable=invalid-name, too-many-branches
        """
        Calculate asset allocations using HRP algorithm.

        :param asset_names: (list) A list of strings containing the asset names
        :param asset_prices: (pd.Dataframe) A dataframe of historical asset prices (daily close)
                                            indexed by date
        :param asset_returns: (pd.Dataframe/numpy matrix) User supplied matrix of asset returns
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns
        :param distance_matrix: (pd.Dataframe/numpy matrix) User supplied distance matrix
        :param side_weights: (pd.Series/numpy matrix) With asset_names in index and value 1 for Buy, -1 for Sell
                                                      (default 1 for all)
        :param linkage: (string) Type of linkage used for Hierarchical Clustering. Supported strings - ``single``,
                                 ``average``, ``complete``, ``ward``.
        """

        pass

    def plot_clusters(self, assets):
        """
        Plot a dendrogram of the hierarchical clusters.

        :param assets: (list) Asset names in the portfolio
        :return: (dict) Dendrogram
        """


        pass

    def _nan_and_diagonal_checks(matrix, nan_fill_value=0, diagonal_fill_value=None):
        """
        Check for any NaN values in the matrix and discrepancies in the diagonal values.
        :param matrix: (pd.DataFrame) The matrix which needs to be processed.
        :param nan_fill_value: (float) Replacement value for NaNs
        :param diagonal_fill_value: (float) The values to use for filling the diagonal.
        :return: (pd.DataFrame) Processed matrix.
        """

        pass

    @staticmethod
    def _tree_clustering(distance, method='single'):
        """
        Perform the traditional heirarchical tree clustering.

        :param correlation: (np.array) Correlation matrix of the assets
        :param method: (str) The type of clustering to be done
        :return: (np.array) Distance matrix and clusters
        """


        pass

    def _quasi_diagnalization(self, num_assets, curr_index):
        """
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param num_assets: (int) The total number of assets
        :param curr_index: (int) Current index
        :return: (list) The assets rearranged according to hierarchical clustering
        """


        pass

    def _get_seriated_matrix(self, assets, distance, correlation):
        """
        Based on the quasi-diagnalization, reorder the original distance matrix, so that assets within
        the same cluster are grouped together.

        :param assets: (list) Asset names in the portfolio
        :param distance: (pd.Dataframe) Distance values between asset returns
        :param correlation: (pd.Dataframe) Correlations between asset returns
        :return: (np.array) Re-arranged distance matrix based on tree clusters
        """

        pass

    def _build_long_short_portfolio(self, side_weights):
        """
        Adjust weights according the shorting constraints specified.

        :param side_weights: (pd.Series/numpy matrix) With asset_names in index and value 1 for Buy, -1 for Sell
                                                      (default 1 for all)
        """

        pass

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        """
        Calculate the inverse variance weight allocations.

        :param covariance: (pd.Dataframe) Covariance matrix of assets
        :return: (list) Inverse variance weight values
        """

        pass

    def _get_cluster_variance(self, covariance, cluster_indices):
        """
        Calculate cluster variance.

        :param covariance: (pd.Dataframe) Covariance matrix of assets
        :param cluster_indices: (list) Asset indices for the cluster
        :return: (float) Variance of the cluster
        """

        pass

    def _recursive_bisection(self, covariance, assets):
        """
        Recursively assign weights to the clusters - ultimately assigning weights to the individual assets.

        :param covariance: (pd.Dataframe) The covariance matrix
        :param assets: (list) Asset names in the portfolio
        """

        pass

    @staticmethod
    def _error_checks(asset_prices, asset_returns, covariance_matrix):
        """
        Perform initial warning checks.

        :param asset_prices: (pd.DataFrame) A dataframe of historical asset prices (daily close)
                                            indexed by date.
        :param asset_returns: (pd.DataFrame/numpy matrix) User supplied matrix of asset returns.
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns
        """

        pass
