"""
Optimal Number of Clusters (ONC Algorithm)
Detection of False Investment Strategies using Unsupervised Learning Methods
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017
"""

from typing import Union

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def _improve_clusters(corr_mat: pd.DataFrame, clusters: dict, top_clusters: dict) -> Union[
        pd.DataFrame, dict, pd.Series]:
    """
    Improve number clusters using silh scores

    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param clusters: (dict) Clusters elements
    :param top_clusters: (dict) Improved clusters elements
    :return: (tuple) [ordered correlation matrix, clusters, silh scores]
    """
    clusters_new, new_idx = {}, []
    for i in clusters.keys():
        clusters_new[len(clusters_new.keys())] = list(clusters[i])

    for i in top_clusters.keys():
        clusters_new[len(clusters_new.keys())] = list(top_clusters[i])

    map(new_idx.extend, clusters_new.values())
    corr_new = corr_mat.loc[new_idx, new_idx]

    dist = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5

    kmeans_labels = np.zeros(len(dist.columns))
    for i in clusters_new:
        idxs = [dist.index.get_loc(k) for k in clusters_new[i]]
        kmeans_labels[idxs] = i

    silh_scores_new = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)
    return corr_new, clusters_new, silh_scores_new


def _cluster_kmeans_base(corr_mat: pd.DataFrame, max_num_clusters: int = 10, repeat: int = 10) -> Union[
        pd.DataFrame, dict, pd.Series]:
    """
    Initial clustering step using KMeans.

    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param max_num_clusters: (int) Maximum number of clusters to search for.
    :param repeat: (int) Number of clustering algorithm repetitions.
    :return: (tuple) [ordered correlation matrix, clusters, silh scores]
    """

    # Distance matrix

    # Fill main diagonal of corr matrix with 1s to avoid elements being close to 1 with e-16.
    # As this previously caused Errors when taking square root from negative values.
    corr_mat[corr_mat > 1] = 1
    distance = ((1 - corr_mat.fillna(0)) / 2.0) ** 0.5
    silh = pd.Series(dtype='float64')

    # Get optimal num clusters
    for _ in range(repeat):
        for num_clusters in range(2, max_num_clusters + 1):
            kmeans_ = KMeans(n_clusters=num_clusters, n_init=1)
            kmeans_ = kmeans_.fit(distance)
            silh_ = silhouette_samples(distance, kmeans_.labels_)
            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())

            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh = silh_
                kmeans = kmeans_

    # Number of clusters equals to length(kmeans labels)
    new_idx = np.argsort(kmeans.labels_)

    # Reorder rows
    corr1 = corr_mat.iloc[new_idx]
    # Reorder columns
    corr1 = corr1.iloc[:, new_idx]

    # Cluster members
    clusters = {i: corr_mat.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in
                np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=distance.index)

    return corr1, clusters, silh


def _check_improve_clusters(new_tstat_mean: float, mean_redo_tstat: float, old_cluster: tuple,
                            new_cluster: tuple) -> tuple:
    """
    Checks cluster improvement condition based on t-statistic.

    :param new_tstat_mean: (float) T-statistics
    :param mean_redo_tstat: (float) Average t-statistcs for cluster improvement
    :param old_cluster: (tuple) Old cluster correlation matrix, optimized clusters, silh scores
    :param new_cluster: (tuple) New cluster correlation matrix, optimized clusters, silh scores
    :return: (tuple) Cluster
    """

    if new_tstat_mean > mean_redo_tstat:
        return old_cluster
    return new_cluster


def cluster_kmeans_top(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series, bool]:
    """
    Improve the initial clustering by leaving clusters with high scores unchanged and modifying clusters with
    below average scores.

    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param repeat: (int) Number of clustering algorithm repetitions.
    :return: (tuple) [correlation matrix, optimized clusters, silh scores, boolean to rerun ONC]
    """
    # pylint: disable=no-else-return

    max_num_clusters = min(corr_mat.drop_duplicates().shape[0], corr_mat.drop_duplicates().shape[1]) - 1
    corr1, clusters, silh = _cluster_kmeans_base(corr_mat, max_num_clusters=max_num_clusters, repeat=repeat)

    # Get cluster quality scores
    cluster_quality = {i: float('Inf') if np.std(silh[clusters[i]]) == 0 else np.mean(silh[clusters[i]]) /
                          np.std(silh[clusters[i]]) for i in clusters.keys()}
    avg_quality = np.mean(list(cluster_quality.values()))
    redo_clusters = [i for i in cluster_quality.keys() if cluster_quality[i] < avg_quality]

    if len(redo_clusters) <= 2:
        # If 2 or less clusters have a quality rating less than the average then stop.
        return corr1, clusters, silh
    else:
        keys_redo = []
        for i in redo_clusters:
            keys_redo.extend(clusters[i])

        corr_tmp = corr_mat.loc[keys_redo, keys_redo]
        mean_redo_tstat = np.mean([cluster_quality[i] for i in redo_clusters])
        _, top_clusters, _ = cluster_kmeans_top(corr_tmp, repeat=repeat)

        # Make new clusters (improved)
        corr_new, clusters_new, silh_new = _improve_clusters(corr_mat,
                                                             {i: clusters[i] for i in clusters.keys() if
                                                              i not in redo_clusters},
                                                             top_clusters)
        new_tstat_mean = np.mean(
            [np.mean(silh_new[clusters_new[i]]) / np.std(silh_new[clusters_new[i]]) for i in clusters_new])

        return _check_improve_clusters(new_tstat_mean, mean_redo_tstat, (corr1, clusters, silh),
                                       (corr_new, clusters_new, silh_new))


def get_onc_clusters(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series]:
    """
    Optimal Number of Clusters (ONC) algorithm described in the following paper:
    `Marcos Lopez de Prado, Michael J. Lewis, Detection of False Investment Strategies Using Unsupervised
    Learning Methods, 2015 <https://papers.ssrn.com/sol3/abstract_id=3167017>`_;
    The code is based on the code provided by the authors of the paper.

    The algorithm searches for the optimal number of clusters using the correlation matrix of elements as an input.

    The correlation matrix is transformed to a matrix of distances, the K-Means algorithm is applied multiple times
    with a different number of clusters to use. The results are evaluated on the t-statistics of the silhouette scores.

    The output of the algorithm is the reordered correlation matrix (clustered elements are placed close to each other),
    optimal clustering, and silhouette scores.

    :param corr_mat: (pd.DataFrame) Correlation matrix of features
    :param repeat: (int) Number of clustering algorithm repetitions
    :return: (tuple) [correlation matrix, optimized clusters, silh scores]
    """

    return cluster_kmeans_top(corr_mat, repeat)
