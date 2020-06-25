# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram
from scipy.spatial.distance import squareform
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimators
from mlfinlab.portfolio_optimization.risk_metrics import RiskMetrics
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


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

        self.weights = list()
        self.clusters = None
        self.ordered_indices = None
        self.cluster_children = None
        self.optimal_num_clusters = None
        self.returns_estimator = ReturnsEstimators()
        self.risk_estimator = RiskEstimators()
        self.risk_metrics = RiskMetrics()
        self.confidence_level = confidence_level

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

        # Perform error checks
        self._error_checks(asset_prices, asset_returns, risk_measure, covariance_matrix)

        if asset_names is None:
            if asset_prices is not None:
                asset_names = asset_prices.columns
            elif asset_returns is not None and isinstance(asset_returns, pd.DataFrame):
                asset_names = asset_returns.columns
            else:
                raise ValueError("Please provide a list of asset names")

        # Calculate the returns if the user does not supply a returns dataframe
        if asset_returns is None and (risk_measure in {'expected_shortfall', 'conditional_drawdown_risk'} or covariance_matrix is
                                      None or not optimal_num_clusters):
            asset_returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices)
        asset_returns = pd.DataFrame(asset_returns, columns=asset_names)

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            covariance_matrix = asset_returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        # Calculate correlation from covariance matrix
        corr = self.risk_estimator.cov_to_corr(cov)

        # Calculate the optimal number of clusters
        if not optimal_num_clusters:
            self.optimal_num_clusters = self._get_optimal_number_of_clusters(correlation=corr,
                                                                             linkage=linkage,
                                                                             asset_returns=asset_returns)
        else:
            self.optimal_num_clusters = self._check_max_number_of_clusters(num_clusters=optimal_num_clusters,
                                                                           linkage=linkage,
                                                                           correlation=corr)

        # Tree Clustering
        self.clusters, self.cluster_children = self._tree_clustering(correlation=corr,
                                                                     linkage=linkage)

        # Get the flattened order of assets in hierarchical clustering tree
        num_assets = len(asset_names)
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)

        # Recursive Bisection
        self._recursive_bisection(asset_returns=asset_returns,
                                  covariance_matrix=cov,
                                  assets=asset_names,
                                  risk_measure=risk_measure)

    def plot_clusters(self, assets):
        """
        Plot a dendrogram of the hierarchical clusters.

        :param assets: (list) Asset names in the portfolio
        :return: (dict) Dendrogram
        """

        colors = dict()
        for cluster_idx, children in self.cluster_children.items():
            color = self.UniqueColors[cluster_idx]

            for child in children:
                colors[assets[child]] = color
        dendrogram_plot = dendrogram(self.clusters, labels=assets, link_color_func=lambda k: self.UnclusteredColor)
        plot_axis = plt.gca()
        xlbls = plot_axis.get_xmajorticklabels()
        for lbl in xlbls:
            lbl.set_color(colors[lbl.get_text()])
        return dendrogram_plot

    @staticmethod
    def _compute_cluster_inertia(labels, asset_returns):
        """
        Calculate the cluster inertia (within cluster sum-of-squares).

        :param labels: (list) Cluster labels.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :return: (float) Cluster inertia value.
        """

        unique_labels = np.unique(labels)
        inertia = [np.mean(pairwise_distances(asset_returns[:, labels == label])) for label in unique_labels]
        inertia = np.log(np.sum(inertia))
        return inertia

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

        distance_matrix = np.sqrt(2 * (1 - correlation).round(5))
        clusters = scipy_linkage(squareform(distance_matrix.values), method=linkage)
        clustering_inds = fcluster(clusters, num_clusters, criterion='maxclust')
        max_number_of_clusters_possible = max(clustering_inds)
        num_clusters = min(max_number_of_clusters_possible, num_clusters)
        return num_clusters

    def _get_optimal_number_of_clusters(self, correlation, asset_returns, linkage, num_reference_datasets=5):
        """
        Find the optimal number of clusters for hierarchical clustering using the Gap statistic.

        :param correlation: (np.array) Matrix of asset correlations.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param linkage: (str) The type of linkage method to use for clustering.
        :param num_reference_datasets: (int) The number of reference datasets to generate for calculating expected inertia.
        :return: (int) The optimal number of clusters.
        """

        original_distance_matrix = np.sqrt(2 * (1 - correlation).round(5))
        gap_values = []
        num_clusters = 1
        max_number_of_clusters = float("-inf")
        while True:

            # Calculate inertia from original data
            original_clusters = scipy_linkage(squareform(original_distance_matrix), method=linkage)
            original_cluster_assignments = fcluster(original_clusters, num_clusters, criterion='maxclust')
            if max(original_cluster_assignments) == max_number_of_clusters or max(original_cluster_assignments) > 10:
                break
            max_number_of_clusters = max(original_cluster_assignments)
            inertia = self._compute_cluster_inertia(original_cluster_assignments, asset_returns.values)

            # Calculate expected inertia from reference datasets
            expected_inertia = self._calculate_expected_inertia(num_reference_datasets, asset_returns, num_clusters, linkage)

            # Calculate the gap statistic
            gap = expected_inertia - inertia
            gap_values.append(gap)
            num_clusters += 1
        return 1 + np.argmax(gap_values)

    def _calculate_expected_inertia(self, num_reference_datasets, asset_returns, num_clusters, linkage):
        """
        Calculate the expected inertia by generating clusters from a uniform distribution.

        :param num_reference_datasets: (int) The number of reference datasets to generate from the distribution.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param num_clusters: (int) The number of clusters to generate.
        :param linkage: (str) The type of linkage criterion to use for hierarchical clustering.
        :return: (float) The expected inertia from the reference datasets.
        """

        reference_inertias = []
        for _ in range(num_reference_datasets):
            # Generate reference returns from uniform distribution and calculate the distance matrix.
            reference_asset_returns = pd.DataFrame(np.random.rand(*asset_returns.shape))
            reference_correlation = np.array(reference_asset_returns.corr())
            reference_distance_matrix = np.sqrt(2 * (1 - reference_correlation).round(5))

            reference_clusters = scipy_linkage(squareform(reference_distance_matrix), method=linkage)
            reference_cluster_assignments = fcluster(reference_clusters, num_clusters, criterion='maxclust')
            inertia = self._compute_cluster_inertia(reference_cluster_assignments, reference_asset_returns.values)
            reference_inertias.append(inertia)
        return np.mean(reference_inertias)

    def _tree_clustering(self, correlation, linkage):
        """
        Perform agglomerative clustering on the current portfolio.

        :param correlation: (np.array) Matrix of asset correlations.
        :param linkage (str): The type of linkage method to use for clustering.
        :return: (list) Structure of hierarchical tree.
        """

        distance_matrix = np.sqrt(2 * (1 - correlation).round(5))
        clusters = scipy_linkage(squareform(distance_matrix.values), method=linkage)
        clustering_inds = fcluster(clusters, self.optimal_num_clusters, criterion='maxclust')
        cluster_children = {index - 1: [] for index in range(min(clustering_inds), max(clustering_inds) + 1)}
        for index, cluster_index in enumerate(clustering_inds):
            cluster_children[cluster_index - 1].append(index)
        return clusters, cluster_children

    def _quasi_diagnalization(self, num_assets, curr_index):
        """
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param num_assets: (int) The total number of assets.
        :param curr_index: (int) Current index.
        :return: (list) The assets rearranged according to hierarchical clustering.
        """

        if curr_index < num_assets:
            return [curr_index]

        left = int(self.clusters[curr_index - num_assets, 0])
        right = int(self.clusters[curr_index - num_assets, 1])

        return (self._quasi_diagnalization(num_assets, left) + self._quasi_diagnalization(num_assets, right))

    def _recursive_bisection(self, asset_returns, covariance_matrix, assets, risk_measure):
        """
        Recursively assign weights to the clusters - ultimately assigning weights to the individual assets.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param covariance_matrix: (pd.DataFrame) The covariance matrix.
        :param assets: (list) List of asset names in the portfolio.
        :param risk_measure: (str) The metric used for calculating weight allocations.
        """

        num_assets = len(assets)
        self.weights = np.ones(shape=num_assets)
        clusters_contribution = np.ones(shape=self.optimal_num_clusters)
        clusters_weights = np.ones(shape=self.optimal_num_clusters)

        # Calculate the corresponding risk measure for the clusters
        self._calculate_risk_contribution_of_clusters(clusters_contribution,
                                                      risk_measure,
                                                      covariance_matrix,
                                                      asset_returns)

        # Recursive bisection taking into account the dendrogram structure
        for cluster_index in range(self.optimal_num_clusters - 1):

            # Get the left and right cluster ids
            left_cluster_ids, right_cluster_ids = self._get_children_cluster_ids(num_assets=num_assets,
                                                                                 parent_cluster_id=cluster_index)

            # Compute alpha
            left_cluster_contribution = np.sum(clusters_contribution[left_cluster_ids])
            right_cluster_contribution = np.sum(clusters_contribution[right_cluster_ids])
            if risk_measure == 'equal_weighting':
                alloc_factor = 0.5
            else:
                alloc_factor = 1 - left_cluster_contribution / (left_cluster_contribution + right_cluster_contribution)

            # Assign weights to each sub-cluster
            clusters_weights[left_cluster_ids] *= alloc_factor
            clusters_weights[right_cluster_ids] *= 1 - alloc_factor

        # Compute the final weights
        self._calculate_final_portfolio_weights(risk_measure,
                                                clusters_weights,
                                                covariance_matrix,
                                                asset_returns)

        # Assign actual asset names to weight index
        self.weights = pd.DataFrame(self.weights)
        self.weights.index = assets
        self.weights = self.weights.T
        self.weights = self.weights.iloc[:, self.ordered_indices]

    def _calculate_final_portfolio_weights(self, risk_measure, clusters_weights, covariance_matrix, asset_returns):
        """
        Calculate the final asset weights.

        :param risk_measure: (str) The metric used for calculating weight allocations.
        :param clusters_weights: (np.array) The cluster weights calculated using recursive bisection.
        :param covariance_matrix: (pd.DataFrame) The covariance matrix.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        """

        for cluster_index in range(self.optimal_num_clusters):
            cluster_asset_indices = self.cluster_children[cluster_index]

            # Covariance of assets in this cluster
            cluster_covariance = covariance_matrix.iloc[cluster_asset_indices, cluster_asset_indices]

            # Historical returns of assets in this cluster
            cluster_asset_returns = None
            if not asset_returns.empty:
                cluster_asset_returns = asset_returns.iloc[:, cluster_asset_indices]

            parity_weights = self._calculate_naive_risk_parity(cluster_index=cluster_index,
                                                               risk_measure=risk_measure,
                                                               covariance=cluster_covariance,
                                                               asset_returns=cluster_asset_returns)
            self.weights[cluster_asset_indices] = parity_weights * clusters_weights[cluster_index]

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

        if risk_measure == 'equal_weighting':
            num_assets_in_cluster = len(self.cluster_children[cluster_index])
            return np.ones(num_assets_in_cluster) * 1/num_assets_in_cluster
        elif risk_measure in {'variance', 'standard_deviation'}:
            return self._get_inverse_variance_weights(covariance)
        elif risk_measure == 'expected_shortfall':
            return self._get_inverse_CVaR_weights(asset_returns)
        return self._get_inverse_CDaR_weights(asset_returns)

    def _calculate_risk_contribution_of_clusters(self, clusters_contribution, risk_measure,
                                                 covariance_matrix, asset_returns):
        """
        Calculate the risk contribution of clusters based on the allocation metric.

        :param clusters_contribution: (np.array) The risk contribution value of the clusters.
        :param risk_measure: (str) The metric used for calculating weight allocations.
        :param covariance_matrix: (pd.DataFrame) The covariance matrix.
        :param asset_returns: (pd.DataFrame) Historical asset returns.
        """

        for cluster_index in range(self.optimal_num_clusters):
            cluster_asset_indices = self.cluster_children[cluster_index]

            if risk_measure == 'variance':
                clusters_contribution[cluster_index] = self._get_cluster_variance(covariance_matrix,
                                                                                  cluster_asset_indices)
            elif risk_measure == 'standard_deviation':
                clusters_contribution[cluster_index] = np.sqrt(
                    self._get_cluster_variance(covariance_matrix, cluster_asset_indices))
            elif risk_measure == 'expected_shortfall':
                clusters_contribution[cluster_index] = self._get_cluster_expected_shortfall(asset_returns,
                                                                                            cluster_asset_indices)
            elif risk_measure == 'conditional_drawdown_risk':
                clusters_contribution[cluster_index] = self._get_cluster_conditional_drawdown_at_risk(
                    asset_returns=asset_returns,
                    cluster_indices=cluster_asset_indices)

    def _get_children_cluster_ids(self, num_assets, parent_cluster_id):
        """
        Find the left and right children cluster id of the given parent cluster id.

        :param num_assets: (int) The number of assets in the portfolio.
        :param parent_cluster_index: (int) The current parent cluster id.
        :return: (list, list) List of cluster ids to the left and right of the parent cluster in the hierarchical tree.
        """

        left = int(self.clusters[num_assets - 2 - parent_cluster_id, 0])
        right = int(self.clusters[num_assets - 2 - parent_cluster_id, 1])
        left_cluster = self._quasi_diagnalization(num_assets, left)
        right_cluster = self._quasi_diagnalization(num_assets, right)

        left_cluster_ids = []
        right_cluster_ids = []
        for id_cluster, cluster in self.cluster_children.items():
            if sorted(self._intersection(left_cluster, cluster)) == sorted(cluster):
                left_cluster_ids.append(id_cluster)
            if sorted(self._intersection(right_cluster, cluster)) == sorted(cluster):
                right_cluster_ids.append(id_cluster)

        return left_cluster_ids, right_cluster_ids

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        """
        Calculate inverse variance weight allocations.

        :param covariance: (pd.DataFrame) Covariance matrix of assets.
        :return: (np.array) Inverse variance weight values.
        """

        inv_diag = 1 / np.diag(covariance.values)
        parity_weights = inv_diag * (1 / np.sum(inv_diag))
        return parity_weights

    def _get_inverse_CVaR_weights(self, asset_returns):
        # pylint: disable=invalid-name
        """
        Calculate inverse CVaR weight allocations.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :return: (np.array) Inverse CVaR weight values.
        """

        parity_weights = []
        for asset_index in range(asset_returns.shape[1]):
            returns = asset_returns.iloc[:, asset_index]
            cvar = self.risk_metrics.calculate_expected_shortfall(returns=returns,
                                                                  confidence_level=self.confidence_level)
            parity_weights.append(cvar)

        parity_weights = np.array(parity_weights)
        parity_weights = 1 / parity_weights
        parity_weights = parity_weights * (1 / np.sum(parity_weights))
        return parity_weights

    def _get_inverse_CDaR_weights(self, asset_returns):
        # pylint: disable=invalid-name
        """
        Calculate inverse CDaR weight allocations.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :return: (np.array) Inverse CDaR weight values.
        """

        parity_weights = []
        for asset_index in range(asset_returns.shape[1]):
            returns = asset_returns.iloc[:, asset_index]
            cdar = self.risk_metrics.calculate_conditional_drawdown_risk(returns=returns,
                                                                         confidence_level=self.confidence_level)
            parity_weights.append(cdar)

        parity_weights = np.array(parity_weights)
        parity_weights = 1 / parity_weights
        parity_weights = parity_weights * (1 / np.sum(parity_weights))
        return parity_weights

    def _get_cluster_variance(self, covariance, cluster_indices):
        """
        Calculate cluster variance.

        :param covariance: (pd.DataFrame) Covariance matrix of asset returns.
        :param cluster_indices: (list) List of asset indices for the cluster.
        :return: (float) Variance of the cluster.
        """

        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_weights = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_weights)
        return cluster_variance

    def _get_cluster_expected_shortfall(self, asset_returns, cluster_indices):
        """
        Calculate cluster expected shortfall.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param cluster_indices: (list) List of asset indices for the cluster.
        :return: (float) Expected shortfall of the cluster.
        """

        cluster_asset_returns = asset_returns.iloc[:, cluster_indices]
        parity_weights = self._get_inverse_CVaR_weights(cluster_asset_returns)
        portfolio_returns = cluster_asset_returns @ parity_weights
        cluster_expected_shortfall = self.risk_metrics.calculate_expected_shortfall(returns=portfolio_returns,
                                                                                    confidence_level=self.confidence_level)
        return cluster_expected_shortfall

    def _get_cluster_conditional_drawdown_at_risk(self, asset_returns, cluster_indices):
        """
        Calculate cluster conditional drawdown at risk.

        :param asset_returns: (pd.DataFrame) Historical asset returns.
        :param cluster_indices: (list) List of asset indices for the cluster.
        :return: (float) CDD of the cluster.
        """

        cluster_asset_returns = asset_returns.iloc[:, cluster_indices]
        parity_weights = self._get_inverse_CDaR_weights(cluster_asset_returns)
        portfolio_returns = cluster_asset_returns @ parity_weights
        cluster_conditional_drawdown = self.risk_metrics.calculate_conditional_drawdown_risk(returns=portfolio_returns,
                                                                                             confidence_level=self.confidence_level)
        return cluster_conditional_drawdown

    @staticmethod
    def _intersection(list1, list2):
        """
        Calculate the intersection of two lists

        :param list1: (list) The first list of items.
        :param list2: (list) The second list of items.
        :return: (list) List containing the intersection of the input lists.
        """

        return list(set(list1) & set(list2))

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

        if asset_prices is None and asset_returns is None and covariance_matrix is None:
            raise ValueError("You need to supply either raw prices or returns or covariance matrix")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        if risk_measure not in \
                {'variance', 'standard_deviation', 'equal_weighting', 'expected_shortfall',
                 'conditional_drawdown_risk'}:
            raise ValueError("Unknown allocation metric specified. Supported metrics are - variance, "
                             "standard_deviation, equal_weighting, expected_shortfall, "
                             "conditional_drawdown_risk")
