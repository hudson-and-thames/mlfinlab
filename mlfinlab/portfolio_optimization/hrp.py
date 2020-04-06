# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import OAS
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation
from mlfinlab.portfolio_optimization.risk_metrics import RiskMetrics


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
        self.seriated_correlations = None
        self.seriated_distances = None
        self.ordered_indices = None
        self.clusters = None
        self.returns_estimator = ReturnsEstimation()
        self.risk_metrics = RiskMetrics()

    def allocate(self,
                 asset_names=None,
                 asset_prices=None,
                 asset_returns=None,
                 covariance_matrix=None,
                 resample_by=None,
                 use_shrinkage=False):
        # pylint: disable=invalid-name, too-many-branches
        """
        Calculate asset allocations using HRP algorithm.

        :param asset_names: (list) a list of strings containing the asset names
        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
                                            indexed by date
        :param asset_returns: (pd.Dataframe/numpy matrix) user supplied matrix of asset returns
        :param covariance_matrix: (pd.Dataframe/numpy matrix) user supplied covariance matrix of asset returns
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        :param use_shrinkage: (Boolean) specifies whether to shrink the covariances
        """

        if asset_prices is None and asset_returns is None and covariance_matrix is None:
            raise ValueError("You need to supply either raw prices or returns or a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        if asset_names is None:
            if asset_prices is not None:
                asset_names = asset_prices.columns
            elif asset_returns is not None and isinstance(asset_returns, pd.DataFrame):
                asset_names = asset_returns.columns
            else:
                raise ValueError("Please provide a list of asset names")

        # Calculate the returns if the user does not supply a returns dataframe
        if asset_returns is None and covariance_matrix is None:
            asset_returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices, resample_by=resample_by)
        asset_returns = pd.DataFrame(asset_returns, columns=asset_names)

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            if use_shrinkage:
                covariance_matrix = self._shrink_covariance(asset_returns=asset_returns)
            else:
                covariance_matrix = asset_returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        # Calculate correlation from covariance matrix
        corr = self._cov2corr(covariance=cov)

        # Step-1: Tree Clustering
        distances, self.clusters = self._tree_clustering(correlation=corr)

        # Step-2: Quasi Diagnalization
        num_assets = len(asset_names)
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)
        self.seriated_distances, self.seriated_correlations = self._get_seriated_matrix(assets=asset_names,
                                                                                        distances=distances,
                                                                                        correlations=corr)

        # Step-3: Recursive Bisection
        self._recursive_bisection(covariance=cov, assets=asset_names)

    @staticmethod
    def _tree_clustering(correlation, method='single'):
        """
        Perform the traditional heirarchical tree clustering.

        :param correlation: (np.array) correlation matrix of the assets
        :param method: (str) the type of clustering to be done
        :return: distance matrix and clusters
        """

        distances = np.sqrt((1 - correlation).round(5) / 2)
        clusters = linkage(squareform(distances.values), method=method)
        return distances, clusters

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

    def _get_seriated_matrix(self, assets, distances, correlations):
        """
        Based on the quasi-diagnalization, reorder the original distance matrix, so that assets within
        the same cluster are grouped together.

        :param assets: (list) list of asset names in the portfolio
        :param distances: (pd.Dataframe) distance values between asset returns
        :param correlations: (pd.Dataframe) correlations between asset returns
        :return: (np.array) re-arranged distance matrix based on tree clusters
        """

        ordering = assets[self.ordered_indices]
        seriated_distances = distances.loc[ordering, ordering]
        seriated_correlations = correlations.loc[ordering, ordering]
        return seriated_distances, seriated_correlations

    @staticmethod
    def _get_inverse_variance_weights(covariance):
        """
        Calculate the inverse variance weight allocations.

        :param covariance: (pd.Dataframe) covariance matrix of assets
        :return: (list) inverse variance weight values
        """

        inv_diag = 1 / np.diag(covariance.values)
        parity_w = inv_diag * (1 / np.sum(inv_diag))
        return parity_w

    def _get_cluster_variance(self, covariance, cluster_indices):
        """
        Calculate cluster variance.

        :param covariance: (pd.Dataframe) covariance matrix of assets
        :param cluster_indices: (list) list of asset indices for the cluster
        :return: (float) variance of the cluster
        """

        cluster_covariance = covariance.iloc[cluster_indices, cluster_indices]
        parity_w = self._get_inverse_variance_weights(cluster_covariance)
        cluster_variance = self.risk_metrics.calculate_variance(covariance=cluster_covariance, weights=parity_w)
        return cluster_variance

    def _recursive_bisection(self, covariance, assets):
        """
        Recursively assign weights to the clusters - ultimately assigning weights to the inidividual assets.

        :param covariance: (pd.Dataframe) the covariance matrix
        :param assets: (list) list of asset names in the portfolio
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
        self.weights = self.weights.T

    def plot_clusters(self, assets):
        """
        Plot a dendrogram of the hierarchical clusters.

        :param assets: (list) list of asset names in the portfolio
        """

        dendrogram_plot = dendrogram(self.clusters, labels=assets)
        return dendrogram_plot

    @staticmethod
    def _shrink_covariance(asset_returns):
        """
        Regularise/Shrink the asset covariances.

        :param asset_returns: (pd.Dataframe) asset returns
        :return: (pd.Dataframe) shrinked asset returns covariances
        """

        oas = OAS()
        oas.fit(asset_returns)
        shrinked_covariance = oas.covariance_
        return shrinked_covariance

    @staticmethod
    def _cov2corr(covariance):
        """
        Calculate the correlations from asset returns covariance matrix.

        :param covariance: (pd.Dataframe) asset returns covariances
        :return: (pd.Dataframe) correlations between asset returns
        """

        d_matrix = np.zeros_like(covariance)
        diagnoal_sqrt = np.sqrt(np.diag(covariance))
        np.fill_diagonal(d_matrix, diagnoal_sqrt)
        d_inv = np.linalg.inv(d_matrix)
        corr = np.dot(np.dot(d_inv, covariance), d_inv)
        corr = pd.DataFrame(corr, index=covariance.columns, columns=covariance.columns)
        return corr
