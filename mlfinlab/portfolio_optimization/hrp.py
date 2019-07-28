import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


class HierarchicalRiskParity:

    def __init__(self):
        return

    def _tree_clustering(self, correlation, method = 'single'):
        '''
        Perform the traditional heirarchical tree clustering

        :param correlation: (np.array) correlation matrix of the assets
        :param method: (str) the type of clustering to be done
        :return: distance matrix and clusters
        '''

        distances = np.sqrt((1 - correlation) / 2)
        clusters = linkage(squareform(distances.values), method = method)
        return distances, clusters

    def _quasi_diagnalization(self, N, curr_index):
        '''
        Rearrange the assets to reorder them according to hierarchical tree clustering order.

        :param N: (int) index of element in the cluster list
        :param curr_index: (int) current index
        :return: (list) the assets rearranged according to hierarchical clustering
        '''

        if curr_index < N:
            return [curr_index]

        left = int(self.clusters[curr_index - N, 0])
        right = int(self.clusters[curr_index - N, 1])

        return (self._quasi_diagnalization(N, left) + self._quasi_diagnalization(N, right))

    def _get_seriated_matrix(self, N, ordered_indices):
        '''
        Based on the quasi-diagnalization, reorder the original distance matrix, so that assets within
        the same cluster are grouped together.

        :param N:
        :param ordered_indices:
        :return:
        '''

        seriated_dist = np.zeros((N, N))
        a, b = np.triu_indices(N, k = 1)
        seriated_dist[a, b] = self.distances[[ordered_indices[i] for i in a], [ordered_indices[j] for j in b]]
        seriated_dist[b, a] = seriated_dist[a, b]
        return seriated_dist

    def _recursive_bisection(self, covariances, ordered_indices):
        '''
        Recursively assign weights to the clusters - ultimately assigning weights to the inidividual assets

        :param covariances: (np.array) the covariance matrix
        :param ordered_indices: (list) asset list reordered according to tree clustering
        '''

        self.weights = pd.Series(1, index = ordered_indices)
        clustered_alphas = [ordered_indices]

        while len(clustered_alphas) > 0:
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

    def plot_clusters(self, height = 10, width = 10):
        '''
        Plot a dendrogram of the hierarchical clusters

        :param height: (int) height of the plot
        :param width: (int) width of the plot
        '''

        plt.figure(figsize = (width, height))
        dendrogram(self.clusters)
        plt.show()

    def allocate(self, asset_prices):
        '''
        Calculate asset allocations using HRP algorithm

        :param asset_prices: (pd.Dataframe/np.array) the matrix of historical asset prices (daily close)
        '''

        if type(asset_prices) != pd.DataFrame:
            asset_prices = pd.DataFrame(asset_prices)

        N = asset_prices.shape[1]
        cov, corr = asset_prices.cov(), asset_prices.corr()

        # Step-1: Tree Clustering
        self.distances, self.clusters = self._tree_clustering(correlation = corr)

        # Step-2: Quasi Diagnalization
        ordered_indices = self._quasi_diagnalization(N, 2*N - 2)
        self.seriated_distances = self._get_seriated_matrix(N = N, ordered_indices = ordered_indices)

        # Step-3: Recursive Bisection
        self._recursive_bisection(covariances = cov, ordered_indices = ordered_indices)