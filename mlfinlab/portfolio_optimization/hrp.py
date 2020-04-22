# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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
                 distance_matrix=None,
                 nb_clusters=None,
                 side_weights=None,
                 linkage_method='single',
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
        :param distance_matrix: (pd.Dataframe/numpy matrix) user supplied distance matrix
        :param nb_clusters: (int/float) number of clusters used for Recursive Bisection
        :param side_weights: (pd.Series/numpy matrix) with asset_names in index and value 1 for Buy, -1 for Sell
                                (default 1 for all)
        :param linkage: (string) type of linkage used for Hierarchical Clustering ex: single, average, complete...
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        :param use_shrinkage: (Boolean) specifies whether to shrink the covariances
        """

        if asset_prices is None and asset_returns is None and covariance_matrix is None:
            raise ValueError(
                "You need to supply either raw prices or returns or a covariance matrix of asset returns")

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

        if nb_clusters is not None:
            if not isinstance(nb_clusters, int) and not isinstance(nb_clusters, float):
                raise ValueError("nb_clusters must be an integer or float")

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
        covariance_matrix = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        # Calculate correlation and distance from covariance matrix
        if distance_matrix is None:
            correlation_matrix = self._cov2corr(covariance=covariance_matrix)
            distance_matrix = np.sqrt((1 - correlation_matrix).round(5) / 2)
        distance_matrix = pd.DataFrame(distance_matrix, index=asset_names, columns=asset_names)

        # Step-1: Tree Clustering
        self.clusters = self._tree_clustering(distance=distance_matrix, method=linkage_method)

        # Step-2: Quasi Diagnalization
        num_assets = len(asset_names)
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)
        self.seriated_distances = self._get_seriated_matrix(assets=asset_names, distance=distance_matrix)

        if side_weights is None:
            side_weights = pd.Series([1] * num_assets, index=asset_names)
        side_weights = pd.Series(side_weights, index=asset_names)

        if nb_clusters is None:
            nb_clusters = num_assets

        # Step-3: Recursive Bisection
        self._recursive_bisection(covariance=covariance_matrix, assets=asset_names, nb_clusters=nb_clusters,
                                  side_weights=side_weights)

    @staticmethod
    def _tree_clustering(distance, method='single'):
        """
        Perform the traditional heirarchical tree clustering.

        :param correlation: (np.array) correlation matrix of the assets
        :param method: (str) the type of clustering to be done
        :return: distance matrix and clusters
        """
        clusters = linkage(squareform(distance.values), method=method)
        return clusters

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

    def _get_seriated_matrix(self, assets, distance):
        """
        Based on the quasi-diagnalization, reorder the original distance matrix, so that assets within
        the same cluster are grouped together.

        :param assets: (list) list of asset names in the portfolio
        :param distance: (pd.Dataframe) distance values between asset returns
        :return: (np.array) re-arranged distance matrix based on tree clusters
        """

        ordering = assets[self.ordered_indices]
        seriated_distances = distance.loc[ordering, ordering]
        return seriated_distances

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

    def _recursive_bisection(self, covariance, assets, nb_clusters, side_weights):
        """
        Recursively assign weights to the clusters - ultimately assigning weights to the inidividual assets.

        :param covariance: (pd.Dataframe) the covariance matrix
        :param assets: (list) list of asset names in the portfolio
        """
        self.weights = pd.Series(1, index=self.ordered_indices)

        # Start by building two clusters
        clusters = pd.Series(fcluster(self.clusters, 2, criterion='maxclust'))
        clusters = clusters.loc[self.ordered_indices]
        left_cluster = list(clusters[clusters == 1].index)
        right_cluster = list(clusters[clusters == 2].index)

        # Initialize the clustered alphas
        clustered_alphas = [left_cluster, right_cluster]

        # Get left and right cluster variances and calculate allocation factor
        left_cluster_variance = self._get_cluster_variance(covariance, left_cluster)
        right_cluster_variance = self._get_cluster_variance(covariance, right_cluster)
        alloc_factor = 1 - left_cluster_variance / (left_cluster_variance + right_cluster_variance)

        # Assign weights to each sub-cluster
        self.weights[left_cluster] *= alloc_factor
        self.weights[right_cluster] *= 1 - alloc_factor

        if nb_clusters >= 3:
            # Loop through the nb of clusters
            for k in range(3, nb_clusters + 1):
                clusters = pd.Series(fcluster(self.clusters, k, criterion='maxclust'))
                clusters = clusters.loc[self.ordered_indices]
                clustered_alphas_ = [list(clusters[clusters == x].index) for x in range(1, k + 1) if
                                     list(clusters[clusters == x].index) != []]
                for idx, cluster in enumerate(clustered_alphas_):
                    if cluster not in clustered_alphas:
                        left_cluster = clustered_alphas_[idx]
                        right_cluster = clustered_alphas_[idx + 1]
                        break
                clustered_alphas = clustered_alphas_

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
        self.weights /= self.weights.sum()

        # Build Long/Short portfolio if needed
        short_ptf = side_weights[side_weights == -1].index
        buy_ptf = side_weights[side_weights == 1].index
        if not short_ptf.empty:
            # Short half size
            self.weights.loc[short_ptf] /= self.weights.loc[short_ptf].sum().values[0]
            self.weights.loc[short_ptf] *= -0.5
            # Buy other half
            self.weights.loc[buy_ptf] /= self.weights.loc[buy_ptf].sum().values[0]
            self.weights.loc[buy_ptf] *= 0.5
        self.weights = self.weights.T

    def plot_clusters(self, max_nb_clusters):
        """
        Plot a dendrogram of the hierarchical clusters.
        :param max_nb_clusters: (int) number of cluster allowing to adjust to deepth of dendogram
        """
        return self._fancy_dendrogram(self.clusters,
                                      truncate_mode='lastp',
                                      p=max_nb_clusters,
                                      leaf_rotation=90.,
                                      leaf_font_size=12.,
                                      show_contracted=True,
                                      annotate_above=10,  # useful in small plots so annotations don't overlap
                                      )

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

    @staticmethod
    def _fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata


