'''
This module implements the HRP algorithm mentioned in the following paper:
`López de Prado, Marcos, Building Diversified Portfolios that Outperform Out-of-Sample (May 23, 2016).
Journal of Portfolio Management, 2016 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_;
The code is reproduced with modification from his book: Advances in Financial Machine Learning, Chp-16
'''

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import OAS
import matplotlib

matplotlib.use('Agg')


class HierarchicalRiskParity:
    '''
    The HRP algorithm is a robust algorithm which tries to overcome the limitations of the CLA algorithm. It has three
    important steps - hierarchical tree clustering, quasi diagnalisation and recursive bisection. Non-inversion of
    covariance matrix makes HRP a very stable algorithm and insensitive to small changes in covariances.
    '''

    def __init__(self):
        self.weights = list()
        self.seriated_correlations = None
        self.seriated_distances = None
        self.ordered_indices = None
        self.clusters = None

    @staticmethod
    def _tree_clustering(correlation, method='single'):
        '''
        Perform the traditional heirarchical tree clustering

        :param correlation: (np.array) correlation matrix of the assets
        :param method: (str) the type of clustering to be done
        :return: distance matrix and clusters
        '''

        distances = np.sqrt((1 - correlation).round(5) / 2)
        clusters = linkage(squareform(distances.values), method=method)
        return distances, clusters

    def _quasi_diagnalization(self, num_assets, curr_index):
        '''
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param num_assets: (int) the total number of assets
        :param curr_index: (int) current index
        :return: (list) the assets rearranged according to hierarchical clustering
        '''

        if curr_index < num_assets:
            return [curr_index]

        left = int(self.clusters[curr_index - num_assets, 0])
        right = int(self.clusters[curr_index - num_assets, 1])

        return (self._quasi_diagnalization(num_assets, left) + self._quasi_diagnalization(num_assets, right))

    def _get_seriated_matrix(self, assets, distances, correlations):
        '''
        Based on the quasi-diagnalization, reorder the original distance matrix, so that assets within
        the same cluster are grouped together.

        :param assets: (list) list of asset names in the portfolio
        :param distances: (pd.Dataframe) distance values between asset returns
        :param correlations: (pd.Dataframe) correlations between asset returns
        :return: (np.array) re-arranged distance matrix based on tree clusters
        '''

        ordering = assets[self.ordered_indices]
        seriated_distances = distances.loc[ordering, ordering]
        seriated_correlations = correlations.loc[ordering, ordering]
        return seriated_distances, seriated_correlations

    def _recursive_bisection(self, covariances, assets):
        '''
        Recursively assign weights to the clusters - ultimately assigning weights to the inidividual assets

        :param covariances: (np.array) the covariance matrix
        :param assets: (list) list of asset names in the portfolio
        '''

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

                # Get left cluster variance
                left_subcovar = covariances.iloc[left_cluster, left_cluster]
                inv_diag = 1 / np.diag(left_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

                # Get right cluster variance
                right_subcovar = covariances.iloc[right_cluster, right_cluster]
                inv_diag = 1 / np.diag(right_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

                # Calculate allocation factor and weights
                alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)
                self.weights[left_cluster] *= alloc_factor
                self.weights[right_cluster] *= 1 - alloc_factor

        # Assign actual asset values to weight index
        self.weights.index = assets[self.ordered_indices]
        self.weights = pd.DataFrame(self.weights)
        self.weights = self.weights.T

    def plot_clusters(self, assets):
        '''
        Plot a dendrogram of the hierarchical clusters

        :param assets: (list) list of asset names in the portfolio
        '''

        dendrogram_plot = dendrogram(self.clusters, labels=assets)
        return dendrogram_plot

    @staticmethod
    def _calculate_returns(asset_prices, resample_by):
        '''
        Calculate the annualised mean historical returns from asset price data

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        :return: (pd.Dataframe) stock returns
        '''

        asset_prices = asset_prices.resample(resample_by).last()
        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        return asset_returns

    @staticmethod
    def _shrink_covariance(covariance):
        '''
        Regularise/Shrink the asset covariances

        :param covariance: (pd.Dataframe) asset returns covariances
        :return: (pd.Dataframe) shrinked asset returns covariances
        '''

        oas = OAS()
        oas.fit(covariance)
        shrinked_covariance = oas.covariance_
        return pd.DataFrame(shrinked_covariance, index=covariance.columns, columns=covariance.columns)

    @staticmethod
    def _cov2corr(covariance):
        '''
        Calculate the correlations from asset returns covariance matrix

        :param covariance: (pd.Dataframe) asset returns covariances
        :return: (pd.Dataframe) correlations between asset returns
        '''

        d_matrix = np.zeros_like(covariance)
        diagnoal_sqrt = np.sqrt(np.diag(covariance))
        np.fill_diagonal(d_matrix, diagnoal_sqrt)
        d_inv = np.linalg.inv(d_matrix)
        corr = np.dot(np.dot(d_inv, covariance), d_inv)
        corr = pd.DataFrame(corr, index=covariance.columns, columns=covariance.columns)
        return corr

    def allocate(self, asset_prices, resample_by='B', use_shrinkage=False):
        '''
        Calculate asset allocations using HRP algorithm

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
                                            indexed by date
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                          'B' meaning daily business days which is equivalent to no resampling
        :param use_shrinkage: (Boolean) specifies whether to shrink the covariances
        '''

        if not isinstance(asset_prices, pd.DataFrame):
            raise ValueError("Asset prices matrix must be a dataframe")
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date.")

        # Calculate the returns
        asset_returns = self._calculate_returns(asset_prices, resample_by=resample_by)

        num_assets = asset_returns.shape[1]
        assets = asset_returns.columns

        # Covariance and correlation
        cov = asset_returns.cov()
        if use_shrinkage:
            cov = self._shrink_covariance(covariance=cov)
        corr = self._cov2corr(covariance=cov)

        # Step-1: Tree Clustering
        distances, self.clusters = self._tree_clustering(correlation=corr)

        # Step-2: Quasi Diagnalization
        self.ordered_indices = self._quasi_diagnalization(num_assets, 2 * num_assets - 2)
        self.seriated_distances, self.seriated_correlations = self._get_seriated_matrix(assets=assets,
                                                                                        distances=distances,
                                                                                        correlations=corr)

        # Step-3: Recursive Bisection
        self._recursive_bisection(covariances=cov, assets=assets)
