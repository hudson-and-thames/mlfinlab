# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram
from scipy.spatial.distance import squareform
from mlfinlab.portfolio_optimization.estimators import ReturnsEstimators
from mlfinlab.portfolio_optimization.utils import RiskMetrics
from mlfinlab.portfolio_optimization.estimators import RiskEstimators


class HierarchicalEqualRiskContribution:
    """
    This class implements the Hierarchical Equal Risk Contribution (HERC) algorithm and it's extended components mentioned in the
    following papers: `Raffinot, Thomas, The Hierarchical Equal Risk Contribution Portfolio (August 23,
    2018). <https://ssrn.com/abstract=3237540>`_; and `Raffinot, Thomas, Hierarchical Clustering Based Asset Allocation (May 2017)
    <https://ssrn.com/abstract=2840729>`_;

    While the vanilla Hierarchical Risk Parity algorithm uses only the variance as a risk measure for assigning weights, the HERC
    algorithm proposed by Raffinot, allows investors to use other risk metrics like Standard Deviation, Expected Shortfall and
    Conditional Drawdown at Risk.
    """

    UniqueColors = ['darkred', 'deepskyblue', 'springgreen', 'darkorange', 'deeppink', 'slateblue', 'navy', 'blueviolet',
                    'pink', 'darkslategray']
    UnclusteredColor = "#808080"
    def __init__(self, confidence_level=0.05):
        """
        Initialise.

        :param confidence_level: (float) The confidence level (alpha) used for calculating expected shortfall and conditional
                                         drawdown at risk.
        """

        pass

    def allocate(self, asset_names=None, asset_prices=None, asset_returns=None, covariance_matrix=None,
                 risk_measure='equal_weighting', linkage='ward', optimal_num_clusters=None):
        # pylint: disable=too-many-branches
        """
        Calculate asset allocations using the Hierarchical Equal Risk Contribution algorithm.

        :param asset_names: (list) A list of strings containing the asset names.
        :param asset_prices: (pd.DataFrame) A dataframe of historical asset prices (daily close)
                                            indexed by date.
        :param asset_returns: (pd.DataFrame/numpy matrix) User supplied matrix of asset returns.
        :param covariance_matrix: (pd.DataFrame/numpy matrix) User supplied covariance matrix of asset returns.
        :param risk_measure: (str) The metric used for calculating weight allocations. Supported strings - ``equal_weighting``,
                                   ``variance``, ``standard_deviation``, ``expected_shortfall``, ``conditional_drawdown_risk``.
        :param linkage: (str) The type of linkage method to use for clustering. Supported strings - ``single``, ``average``,
                              ``complete``, ``ward``.
        :param optimal_num_clusters: (int) Optimal number of clusters for hierarchical clustering.
        """

        pass

    def plot_clusters(self, assets):
        """
        Plot a dendrogram of the hierarchical clusters.

        :param assets: (list) Asset names in the portfolio
        :return: (dict) Dendrogram
        """

        pass

    @staticmethod
    def _compute_cluster_inertia(labels, asset_returns):
        """
        Calculate the cluster inertia (within cluster sum-of-squares).

        :param labels: (list) Cluster labels.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :return: (float) Cluster inertia value.
        """

        pass

    @staticmethod
    def _check_max_number_of_clusters(num_clusters, linkage, correlation):
        """
        In some cases, the optimal number of clusters value given by the users is greater than the maximum number of clusters
        possible with the given data. This function checks this and assigns the proper value to the number of clusters when the
        given value exceeds maximum possible clusters.

        :param num_clusters: (int) The number of clusters.
        :param linkage (str): The type of linkage method to use for clustering.
        :param correlation: (np.array) Matrix of asset correlations.
        :return: (int) New value for number of clusters.
        """
        pass

    def _get_optimal_number_of_clusters(self, correlation, asset_returns, linkage, num_reference_datasets=5):
        """
        Find the optimal number of clusters for hierarchical clustering using the Gap statistic.

        :param correlation: (np.array) Matrix of asset correlations.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param linkage: (str) The type of linkage method to use for clustering.
        :param num_reference_datasets: (int) The number of reference datasets to generate for calculating expected inertia.
        :return: (int) The optimal number of clusters.
        """

        pass

    def _calculate_expected_inertia(self, num_reference_datasets, asset_returns, num_clusters, linkage):
        """
        Calculate the expected inertia by generating clusters from a uniform distribution.

        :param num_reference_datasets: (int) The number of reference datasets to generate from the distribution.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param num_clusters: (int) The number of clusters to generate.
        :param linkage: (str) The type of linkage criterion to use for hierarchical clustering.
        :return: (float) The expected inertia from the reference datasets.
        """

        pass

    def _tree_clustering(self, correlation, linkage):
        """
        Perform agglomerative clustering on the current portfolio.

        :param correlation: (np.array) Matrix of asset correlations.
        :param linkage (str): The type of linkage method to use for clustering.
        :return: (list) Structure of hierarchical tree.
        """

        pass

    def _quasi_diagnalization(self, num_assets, curr_index):
        """
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param num_assets: (int) The total number of assets.
        :param curr_index: (int) Current index.
        :return: (list) The assets rearranged according to hierarchical clustering.
        """

        pass

    def _recursive_bisection(self, asset_returns, covariance_matrix, assets, risk_measure):
        """
        Recursively assign weights to the clusters - ultimately assigning weights to the individual assets.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param covariance_matrix: (pd.DataFrame) The covariance matrix.
        :param assets: (list) List of asset names in the portfolio.
        :param risk_measure: (str) The metric used for calculating weight allocations.
        """

        pass

    def _calculate_final_portfolio_weights(self, risk_measure, clusters_weights, covariance_matrix, asset_returns):
        """
        Calculate the final asset weights.

        :param risk_measure: (str) The metric used for calculating weight allocations.
        :param clusters_weights: (np.array) The cluster weights calculated using recursive bisection.
        :param covariance_matrix: (pd.DataFrame) The covariance matrix.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        """

        pass

    def _calculate_naive_risk_parity(self, cluster_index, risk_measure, covariance, asset_returns):
        # pylint: disable=no-else-return
        """
        Calculate the naive risk parity weights.

        :param cluster_index: (int) Index of the current cluster.
        :param risk_measure: (str) The metric used for calculating weight allocations.
        :param covariance: (pd.DataFrame) The covariance matrix of asset returns.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :return: (np.array) list of risk parity weights for assets in current cluster.
        """

        pass

    def _calculate_risk_contribution_of_clusters(self, clusters_contribution, risk_measure,
                                                 covariance_matrix, asset_returns):
        """
        Calculate the risk contribution of clusters based on the allocation metric.

        :param clusters_contribution: (np.array) The risk contribution value of the clusters.
        :param risk_measure: (str) The metric used for calculating weight allocations.
        :param covariance_matrix: (pd.DataFrame) The covariance matrix.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        """

        pass

    def _get_children_cluster_ids(self, num_assets, parent_cluster_id):
        """
        Find the left and right children cluster id of the given parent cluster id.

        :param num_assets: (int) The number of assets in the portfolio.
        :param parent_cluster_index: (int) The current parent cluster id.
        :return: (list, list) List of cluster ids to the left and right of the parent cluster in the hierarchical tree.
        """

        pass

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        """
        Calculate inverse variance weight allocations.

        :param covariance: (pd.DataFrame) Covariance matrix of assets.
        :return: (np.array) Inverse variance weight values.
        """

        pass

    def _get_inverse_CVaR_weights(self, asset_returns):
        # pylint: disable=invalid-name
        """
        Calculate inverse CVaR weight allocations.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :return: (np.array) Inverse CVaR weight values.
        """

        pass

    def _get_inverse_CDaR_weights(self, asset_returns):
        # pylint: disable=invalid-name
        """
        Calculate inverse CDaR weight allocations.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :return: (np.array) Inverse CDaR weight values.
        """

        pass

    def _get_cluster_variance(self, covariance, cluster_indices):
        """
        Calculate cluster variance.

        :param covariance: (pd.DataFrame) Covariance matrix of asset returns.
        :param cluster_indices: (list) List of asset indices for the cluster.
        :return: (float) Variance of the cluster.
        """

        pass

    def _get_cluster_expected_shortfall(self, asset_returns, cluster_indices):
        """
        Calculate cluster expected shortfall.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param cluster_indices: (list) List of asset indices for the cluster.
        :return: (float) Expected shortfall of the cluster.
        """

        pass

    def _get_cluster_conditional_drawdown_at_risk(self, asset_returns, cluster_indices):
        """
        Calculate cluster conditional drawdown at risk.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param cluster_indices: (list) List of asset indices for the cluster.
        :return: (float) CDD of the cluster.
        """

        pass

    @staticmethod
    def _intersection(list1, list2):
        """
        Calculate the intersection of two lists

        :param list1: (list) The first list of items.
        :param list2: (list) The second list of items.
        :return: (list) List containing the intersection of the input lists.
        """

        pass

    @staticmethod
    def _error_checks(asset_prices, asset_returns, risk_measure, covariance_matrix):
        """
        Perform initial warning checks.

        :param asset_prices: (pd.DataFrame) A dataframe of historical asset prices (daily close)
                                            indexed by date.
        :param asset_returns: (pd.DataFrame/numpy matrix) User supplied matrix of asset returns.
        :param risk_measure: (str) The metric used for calculating weight allocations.
        :param covariance_matrix: (pd.DataFrame/numpy matrix) User supplied covariance matrix of asset returns.
        """


        pass
