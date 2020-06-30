# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram
from scipy.spatial.distance import squareform
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimators
from mlfinlab.portfolio_optimization.risk_metrics import RiskMetrics
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


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
        self.weights = list()
        self.seriated_distances = None
        self.seriated_correlations = None
        self.ordered_indices = None
        self.clusters = None
        self.returns_estimator = ReturnsEstimators()
        self.risk_metrics = RiskMetrics()
        self.risk_estimator = RiskEstimators()

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

        # Perform error checks
        self._error_checks(asset_prices, asset_returns, covariance_matrix)

        if asset_names is None:
            if asset_prices is not None:
                asset_names = asset_prices.columns
            elif asset_returns is not None and isinstance(asset_returns, pd.DataFrame):
                asset_names = asset_returns.columns
            else:
                raise ValueError("Please provide a list of asset names")

        # Calculate the returns if the user does not supply a returns dataframe
        if asset_returns is None and covariance_matrix is None:
            asset_returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices)
        asset_returns = pd.DataFrame(asset_returns, columns=asset_names)

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            covariance_matrix = asset_returns.cov()
        covariance_matrix = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        # Calculate correlation and distance from covariance matrix
        correlation_matrix = self.risk_estimator.cov_to_corr(covariance_matrix)
        if distance_matrix is None:
            distance_matrix = np.sqrt((1 - correlation_matrix).round(5) / 2)
        distance_matrix = pd.DataFrame(distance_matrix, index=asset_names, columns=asset_names)

        # Step-1: Tree Clustering
        self.clusters = self._tree_clustering(distance=distance_matrix, method=linkage)

        # Step-2: Quasi Diagnalization
        num_assets = len(asset_names)
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)
        self.seriated_distances, self.seriated_correlations = self._get_seriated_matrix(assets=asset_names,
                                                                                        distance=distance_matrix,
                                                                                        correlation=correlation_matrix)

        # Step-3: Recursive Bisection
        self._recursive_bisection(covariance=covariance_matrix, assets=asset_names)

        # Build Long/Short portfolio
        if side_weights is None:
            side_weights = pd.Series([1] * num_assets, index=asset_names)
        side_weights = pd.Series(side_weights, index=asset_names)
        self._build_long_short_portfolio(side_weights)

    def plot_clusters(self, assets):
        """
        Plot a dendrogram of the hierarchical clusters.

        :param assets: (list) Asset names in the portfolio
        :return: (dict) Dendrogram
        """

        dendrogram_plot = dendrogram(self.clusters, labels=assets)
        return dendrogram_plot

    @staticmethod
    def _tree_clustering(distance, method='single'):
        """
        Perform the traditional heirarchical tree clustering.

        :param correlation: (np.array) Correlation matrix of the assets
        :param method: (str) The type of clustering to be done
        :return: (np.array) Distance matrix and clusters
        """

        clusters = scipy_linkage(squareform(distance.values), method=method)
        return clusters

    def _quasi_diagnalization(self, num_assets, curr_index):
        """
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param num_assets: (int) The total number of assets
        :param curr_index: (int) Current index
        :return: (list) The assets rearranged according to hierarchical clustering
        """

        if curr_index < num_assets:
            return [curr_index]

        left = int(self.clusters[curr_index - num_assets, 0])
        right = int(self.clusters[curr_index - num_assets, 1])

        return (self._quasi_diagnalization(num_assets, left) + self._quasi_diagnalization(num_assets, right))

    def _get_seriated_matrix(self, assets, distance, correlation):
        """
        Based on the quasi-diagnalization, reorder the original distance matrix, so that assets within
        the same cluster are grouped together.

        :param assets: (list) Asset names in the portfolio
        :param distance: (pd.Dataframe) Distance values between asset returns
        :param correlation: (pd.Dataframe) Correlations between asset returns
        :return: (np.array) Re-arranged distance matrix based on tree clusters
        """

        ordering = assets[self.ordered_indices]
        seriated_distances = distance.loc[ordering, ordering]
        seriated_correlations = correlation.loc[ordering, ordering]
        return seriated_distances, seriated_correlations

    def _build_long_short_portfolio(self, side_weights):
        """
        Adjust weights according the shorting constraints specified.

        :param side_weights: (pd.Series/numpy matrix) With asset_names in index and value 1 for Buy, -1 for Sell
                                                      (default 1 for all)
        """

        short_ptf = side_weights[side_weights == -1].index
        buy_ptf = side_weights[side_weights == 1].index
        if len(short_ptf) > 0:
            # Short half size
            self.weights.loc[short_ptf] /= self.weights.loc[short_ptf].sum().values[0]
            self.weights.loc[short_ptf] *= -0.5

            # Buy other half
            self.weights.loc[buy_ptf] /= self.weights.loc[buy_ptf].sum().values[0]
            self.weights.loc[buy_ptf] *= 0.5
        self.weights = self.weights.T

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        """
        Calculate the inverse variance weight allocations.

        :param covariance: (pd.Dataframe) Covariance matrix of assets
        :return: (list) Inverse variance weight values
        """

        inv_diag = 1 / np.diag(covariance.values)
        parity_w = inv_diag * (1 / np.sum(inv_diag))
        return parity_w

    def _get_cluster_variance(self, covariance, cluster_indices):
        """
        Calculate cluster variance.

        :param covariance: (pd.Dataframe) Covariance matrix of assets
        :param cluster_indices: (list) Asset indices for the cluster
        :return: (float) Variance of the cluster
        """

        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_w)
        return cluster_variance

    def _recursive_bisection(self, covariance, assets):
        """
        Recursively assign weights to the clusters - ultimately assigning weights to the individual assets.

        :param covariance: (pd.Dataframe) The covariance matrix
        :param assets: (list) Asset names in the portfolio
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

                # Get left and right cluster variances and calculate allocation factor
                left_cluster_variance = self._get_cluster_variance(covariance, left_cluster)
                right_cluster_variance = self._get_cluster_variance(covariance, right_cluster)
                alloc_factor = 1 - left_cluster_variance / (left_cluster_variance + right_cluster_variance)

                # Assign weights to each sub-cluster
                self.weights[left_cluster] *= alloc_factor
                self.weights[right_cluster] *= 1 - alloc_factor

        # Assign actual asset values to weight index
        self.weights.index = assets[self.ordered_indices]
        self.weights = pd.DataFrame(self.weights)

    @staticmethod
    def _error_checks(asset_prices, asset_returns, covariance_matrix):
        """
        Perform initial warning checks.

        :param asset_prices: (pd.DataFrame) A dataframe of historical asset prices (daily close)
                                            indexed by date.
        :param asset_returns: (pd.DataFrame/numpy matrix) User supplied matrix of asset returns.
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns
        """


        if asset_prices is None and asset_returns is None and covariance_matrix is None:
            raise ValueError(
                "You need to supply either raw prices or returns or a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")
