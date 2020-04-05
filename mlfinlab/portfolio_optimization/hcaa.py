# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation
from mlfinlab.portfolio_optimization.risk_metrics import RiskMetrics


class HierarchicalClusteringAssetAllocation:
    """
    This class implements the Hierarchical Equal Risk Contribution (HERC) algorithm and it's extended components mentioned in the
    following papers: `Raffinot, Thomas, The Hierarchical Equal Risk Contribution Portfolio (August 23,
    2018). <https://ssrn.com/abstract=3237540>`_; and `Raffinot, Thomas, Hierarchical Clustering Based Asset Allocation (May 2017)
    <https://ssrn.com/abstract=2840729>`_;

    While the vanilla Hierarchical Risk Parity algorithm uses only the variance as a risk measure for assigning weights, the HERC
    algorithm proposed by Raffinot, allows investors to use other risk metrics like Expected Shortfall, Sharpe Ratio and
    Conditional Drawdown. Furthermore, it is flexible enough to be easily extended to include custom risk measures of our own.
    """

    def __init__(self, calculate_expected_returns='mean'):
        """
        Initialise.

        :param calculate_expected_returns: (str) the method to use for calculation of expected returns.
                                        Currently supports "mean" and "exponential"
        """

        self.weights = list()
        self.clusters = None
        self.ordered_indices = None
        self.returns_estimator = ReturnsEstimation()
        self.risk_metrics = RiskMetrics()
        self.calculate_expected_returns = calculate_expected_returns

    def allocate(self,
                 asset_names=None,
                 asset_prices=None,
                 asset_returns=None,
                 covariance_matrix=None,
                 expected_asset_returns=None,
                 allocation_metric='equal_weighting',
                 linkage='average',
                 confidence_level=0.05,
                 optimal_num_clusters=None,
                 resample_by=None):
        # pylint: disable=too-many-arguments
        """
        Calculate asset allocations using the HCAA algorithm.

        :param asset_names: (list) a list of strings containing the asset names
        :param asset_prices: (pd.DataFrame) a dataframe of historical asset prices (daily close)
                                            indexed by date
        :param asset_returns: (pd.DataFrame/numpy matrix) user supplied matrix of asset returns
        :param covariance_matrix: (pd.DataFrame/numpy matrix) user supplied covariance matrix of asset returns
        :param expected_asset_returns: (list) a list of mean asset returns (mu)
        :param allocation_metric: (str) the metric used for calculating weight allocations. Supported strings - "equal_weighting",
                                        "minimum_variance", "minimum_standard_deviation", "sharpe_ratio", "expected_shortfall",
                                        "conditional_drawdown_risk"
        :param linkage: (str) the type of linkage method to use for clustering. Supported strings - "single", "average", "complete"
        :param confidence_level: (float) the confidence level (alpha) used for calculating expected shortfall and conditional
                                         drawdown at risk
        :param optimal_num_clusters: (int) optimal number of clusters for hierarchical clustering
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        """

        # Perform initial checks
        self._perform_checks(asset_prices, asset_returns, covariance_matrix, allocation_metric)

        # Calculate the expected returns if the user does not supply any returns
        if allocation_metric == 'sharpe_ratio' and expected_asset_returns is None:
            if asset_prices is None:
                raise ValueError(
                    "Either provide pre-calculated expected returns or give raw asset prices for inbuilt returns calculation")

            if self.calculate_expected_returns == "mean":
                expected_asset_returns = self.returns_estimator.calculate_mean_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            elif self.calculate_expected_returns == "exponential":
                expected_asset_returns = self.returns_estimator.calculate_exponential_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")

        if asset_names is None:
            if asset_prices is not None:
                asset_names = asset_prices.columns
            elif asset_returns is not None and isinstance(asset_returns, pd.DataFrame):
                asset_names = asset_returns.columns
            else:
                raise ValueError("Please provide a list of asset names")

        # Calculate the returns if the user does not supply a returns dataframe
        if asset_returns is None:
            asset_returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices, resample_by=resample_by)
        asset_returns = pd.DataFrame(asset_returns, columns=asset_names)

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            covariance_matrix = asset_returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        # Calculate correlation from covariance matrix
        corr = self._cov2corr(covariance=cov)

        # Calculate the optimal number of clusters using the Gap statistic
        if not optimal_num_clusters:
            optimal_num_clusters = self._get_optimal_number_of_clusters(correlation=corr,
                                                                        linkage=linkage,
                                                                        asset_returns=asset_returns)

        # Tree Clustering
        self.clusters = self._tree_clustering(correlation=corr, num_clusters=optimal_num_clusters, linkage=linkage)

        # Quasi Diagnalization
        num_assets = len(asset_names)
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)

        # Recursive Bisection
        self._recursive_bisection(expected_asset_returns=expected_asset_returns,
                                  asset_returns=asset_returns,
                                  covariance_matrix=cov,
                                  assets=asset_names,
                                  allocation_metric=allocation_metric,
                                  confidence_level=confidence_level)

    @staticmethod
    def _compute_cluster_inertia(labels, asset_returns):
        """
        Calculate the cluster inertia (within cluster sum-of-squares).

        :param labels: (list) cluster labels
        :param asset_returns: (pd.DataFrame) historical asset returns
        :return: (float) cluster inertia value
        """

        unique_labels = np.unique(labels)
        inertia = [np.mean(pairwise_distances(asset_returns[:, labels == label])) for label in unique_labels]
        inertia = np.log(np.sum(inertia))
        return inertia

    def _get_optimal_number_of_clusters(self,
                                        correlation,
                                        asset_returns,
                                        linkage,
                                        num_reference_datasets=5):
        """
        Find the optimal number of clusters for hierarchical clustering using the Gap statistic.

        :param correlation: (np.array) matrix of asset correlations
        :param asset_returns: (pd.DataFrame) historical asset returns
        :param linkage: (str) the type of linkage method to use for clustering
        :param num_reference_datasets: (int) the number of reference datasets to generate for calculating expected inertia
        :return: (int) the optimal number of clusters
        """

        max_number_of_clusters = min(10, asset_returns.shape[1])
        cluster_func = AgglomerativeClustering(affinity='precomputed', linkage=linkage)
        original_distance_matrix = np.sqrt(2 * (1 - correlation).round(5))
        gap_values = []
        for num_clusters in range(1, max_number_of_clusters + 1):
            cluster_func.n_clusters = num_clusters

            # Calculate expected inertia from reference datasets
            reference_inertias = []
            for _ in range(num_reference_datasets):

                # Generate reference returns from uniform distribution and calculate the distance matrix.
                reference_asset_returns = pd.DataFrame(np.random.rand(*asset_returns.shape))
                reference_correlation = np.array(reference_asset_returns.corr())
                reference_distance_matrix = np.sqrt(2 * (1 - reference_correlation).round(5))

                reference_cluster_assignments = cluster_func.fit_predict(reference_distance_matrix)
                inertia = self._compute_cluster_inertia(reference_cluster_assignments, reference_asset_returns.values)
                reference_inertias.append(inertia)
            expected_inertia = np.mean(reference_inertias)

            # Calculate inertia from original data
            original_cluster_asignments = cluster_func.fit_predict(original_distance_matrix)
            inertia = self._compute_cluster_inertia(original_cluster_asignments, asset_returns.values)

            # Calculate the gap statistic
            gap = expected_inertia - inertia
            gap_values.append(gap)

        return 1 + np.argmax(gap_values)

    @staticmethod
    def _tree_clustering(correlation, num_clusters, linkage):
        """
        Perform agglomerative clustering on the current portfolio.

        :param correlation: (np.array) matrix of asset correlations
        :param num_clusters: (int) the number of clusters
        :param linkage (str): the type of linkage method to use for clustering
        :return: (list) structure of hierarchical tree
        """

        cluster_func = AgglomerativeClustering(n_clusters=num_clusters,
                                               affinity='precomputed',
                                               linkage=linkage)
        distance_matrix = np.sqrt(2 * (1 - correlation).round(5))
        cluster_func.fit(distance_matrix)
        return cluster_func.children_

    def _quasi_diagnalization(self, num_assets, curr_index):
        """
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param num_assets: (int) the total number of assets
        :param curr_index: (int) current index
        :return: (list) the assets rearranged according to hierarchical clustering
        """

        if curr_index < num_assets:
            return [curr_index]

        left = int(self.clusters[curr_index - num_assets, 0])
        right = int(self.clusters[curr_index - num_assets, 1])

        return (self._quasi_diagnalization(num_assets, left) + self._quasi_diagnalization(num_assets, right))

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        """
        Calculate the inverse variance weight allocations.

        :param covariance: (pd.DataFrame) covariance matrix of assets
        :return: (list) inverse variance weight values
        """

        inv_diag = 1 / np.diag(covariance.values)
        parity_w = inv_diag * (1 / np.sum(inv_diag))
        return parity_w

    def _get_cluster_variance(self, covariance, cluster_indices):
        """
        Calculate cluster variance.

        :param covariance: (pd.DataFrame) covariance matrix of assets
        :param cluster_indices: (list) list of asset indices for the cluster
        :return: (float) variance of the cluster
        """

        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_w)
        return cluster_variance

    def _get_cluster_sharpe_ratio(self, expected_asset_returns, covariance, cluster_indices):
        """
        Calculate cluster Sharpe Ratio.

        :param expected_asset_returns: (list) a list of mean asset returns (mu)
        :param covariance: (pd.DataFrame) covariance matrix of assets
        :param cluster_indices: (list) list of asset indices for the cluster
        :return: (float) sharpe ratio of the cluster
        """

        cluster_expected_returns = expected_asset_returns[cluster_indices]
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_w)
        cluster_sharpe_ratio = (parity_w @ cluster_expected_returns) / np.sqrt(cluster_variance)
        return cluster_sharpe_ratio

    def _get_cluster_expected_shortfall(self, asset_returns, covariance, confidence_level, cluster_indices):
        """
        Calculate cluster expected shortfall.

        :param asset_returns: (pd.DataFrame) historical asset returns
        :param covariance: (pd.DataFrame) covariance matrix of assets
        :param confidence_level: (float) the confidence level (alpha)
        :param cluster_indices: (list) list of asset indices for the cluster
        :return: (float) expected shortfall of the cluster
        """

        cluster_asset_returns = asset_returns.iloc[:, cluster_indices]
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        portfolio_returns = cluster_asset_returns @ parity_w
        cluster_expected_shortfall = self.risk_metrics.calculate_expected_shortfall(returns=portfolio_returns,
                                                                                    confidence_level=confidence_level)
        return cluster_expected_shortfall

    def _get_cluster_conditional_drawdown_at_risk(self, asset_returns, covariance, confidence_level, cluster_indices):
        """
        Calculate cluster conditional drawdown at risk.

        :param asset_returns: (pd.DataFrame) historical asset returns
        :param covariance: (pd.DataFrame) covariance matrix of assets
        :param confidence_level: (float) the confidence level (alpha)
        :param cluster_indices: (list) list of asset indices for the cluster
        :return: (float) CDD of the cluster
        """

        cluster_asset_returns = asset_returns.iloc[:, cluster_indices]
        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        portfolio_returns = cluster_asset_returns @ parity_w
        cluster_conditional_drawdown = self.risk_metrics.calculate_conditional_drawdown_risk(returns=portfolio_returns,
                                                                                             confidence_level=confidence_level)
        return cluster_conditional_drawdown

    def _recursive_bisection(self,
                             expected_asset_returns,
                             asset_returns,
                             covariance_matrix,
                             assets,
                             allocation_metric,
                             confidence_level):
        # pylint: disable=bad-continuation, too-many-locals
        """
        Recursively assign weights to the clusters - ultimately assigning weights to the individual assets.

        :param expected_asset_returns: (list) a list of mean asset returns (mu)
        :param asset_returns: (pd.DataFrame) historical asset returns
        :param covariance_matrix: (pd.DataFrame) the covariance matrix
        :param assets: (list) list of asset names in the portfolio
        :param allocation_metric: (str) the metric used for calculating weight allocations
        :param confidence_level: (float) the confidence level (alpha)
        """

        self.weights = pd.Series(1, index=self.ordered_indices)
        clustered_alphas = [self.ordered_indices]

        while clustered_alphas:
            clustered_alphas = [cluster[start:end]
                                for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]

            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]

                # Calculate allocation factor based on the metric
                if allocation_metric == 'minimum_variance':
                    left_cluster_variance = self._get_cluster_variance(covariance_matrix, left_cluster)
                    right_cluster_variance = self._get_cluster_variance(covariance_matrix, right_cluster)
                    alloc_factor = 1 - left_cluster_variance / (left_cluster_variance + right_cluster_variance)
                elif allocation_metric == 'minimum_standard_deviation':
                    left_cluster_sd = np.sqrt(self._get_cluster_variance(covariance_matrix, left_cluster))
                    right_cluster_sd = np.sqrt(self._get_cluster_variance(covariance_matrix, right_cluster))
                    alloc_factor = 1 - left_cluster_sd / (left_cluster_sd + right_cluster_sd)
                elif allocation_metric == 'sharpe_ratio':
                    left_cluster_sharpe_ratio = self._get_cluster_sharpe_ratio(expected_asset_returns,
                                                                               covariance_matrix,
                                                                               left_cluster)
                    right_cluster_sharpe_ratio = self._get_cluster_sharpe_ratio(expected_asset_returns,
                                                                                covariance_matrix,
                                                                                right_cluster)
                    alloc_factor = left_cluster_sharpe_ratio / (left_cluster_sharpe_ratio + right_cluster_sharpe_ratio)

                    if alloc_factor < 0 or alloc_factor > 1:
                        left_cluster_variance = self._get_cluster_variance(covariance_matrix, left_cluster)
                        right_cluster_variance = self._get_cluster_variance(covariance_matrix, right_cluster)
                        alloc_factor = 1 - left_cluster_variance / (left_cluster_variance + right_cluster_variance)
                elif allocation_metric == 'expected_shortfall':
                    left_cluster_expected_shortfall = self._get_cluster_expected_shortfall(asset_returns=asset_returns,
                                                                                           covariance=covariance_matrix,
                                                                                           confidence_level=confidence_level,
                                                                                           cluster_indices=left_cluster)
                    right_cluster_expected_shortfall = self._get_cluster_expected_shortfall(asset_returns=asset_returns,
                                                                                           covariance=covariance_matrix,
                                                                                           confidence_level=confidence_level,
                                                                                           cluster_indices=right_cluster)
                    alloc_factor = \
                        1 - left_cluster_expected_shortfall / (left_cluster_expected_shortfall + right_cluster_expected_shortfall)
                elif allocation_metric == 'conditional_drawdown_risk':
                    left_cluster_conditional_drawdown = self._get_cluster_conditional_drawdown_at_risk(asset_returns=asset_returns,
                                                         covariance=covariance_matrix,
                                                         confidence_level=confidence_level,
                                                         cluster_indices=left_cluster)
                    right_cluster_conditional_drawdown = self._get_cluster_conditional_drawdown_at_risk(asset_returns=asset_returns,
                                                         covariance=covariance_matrix,
                                                         confidence_level=confidence_level,
                                                         cluster_indices=right_cluster)
                    alloc_factor = \
                        1 - left_cluster_conditional_drawdown / (left_cluster_conditional_drawdown + right_cluster_conditional_drawdown)
                else:
                    alloc_factor = 0.5 # equal weighting

                # Assign weights to each sub-cluster
                self.weights[left_cluster] *= alloc_factor
                self.weights[right_cluster] *= 1 - alloc_factor

        # Assign actual asset values to weight index
        self.weights.index = assets[self.ordered_indices]
        self.weights = pd.DataFrame(self.weights)
        self.weights = self.weights.T

    @staticmethod
    def _cov2corr(covariance):
        """
        Calculate the correlations from asset returns covariance matrix.

        :param covariance: (pd.DataFrame) asset returns covariances
        :return: (pd.DataFrame) correlations between asset returns
        """

        d_matrix = np.zeros_like(covariance)
        diagnoal_sqrt = np.sqrt(np.diag(covariance))
        np.fill_diagonal(d_matrix, diagnoal_sqrt)
        d_inv = np.linalg.inv(d_matrix)
        corr = np.dot(np.dot(d_inv, covariance), d_inv)
        corr = pd.DataFrame(corr, index=covariance.columns, columns=covariance.columns)
        return corr

    @staticmethod
    def _perform_checks(asset_prices, asset_returns, covariance_matrix, allocation_metric):
        # pylint: disable=bad-continuation
        """
        Perform initial warning checks.

        :param asset_prices: (pd.DataFrame) a dataframe of historical asset prices (daily close)
                                            indexed by date
        :param asset_returns: (pd.DataFrame/numpy matrix) user supplied matrix of asset returns
        :param covariance_matrix: (pd.DataFrame/numpy matrix) user supplied covariance matrix of asset returns
        :param allocation_metric: (str) the metric used for calculating weight allocations
        :return:
        """

        if asset_prices is None and asset_returns is None and covariance_matrix is None:
            raise ValueError("You need to supply either raw prices or returns or a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        if allocation_metric not in \
                {'minimum_variance', 'minimum_standard_deviation', 'sharpe_ratio',
                 'equal_weighting', 'expected_shortfall', 'conditional_drawdown_risk'}:
            raise ValueError("Unknown allocation metric specified. Supported metrics are - minimum_variance, "
                             "minimum_standard_deviation, sharpe_ratio, equal_weighting, expected_shortfall, "
                             "conditional_drawdown_risk")
