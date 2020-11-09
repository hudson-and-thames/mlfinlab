# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from mlfinlab.portfolio_optimization.estimators.risk_estimators import RiskEstimators


class TheoryImpliedCorrelation:
    """
    This class implements the Theory-Implied Correlation (TIC) algorithm and the correlation matrix distance
    introduced by Herdin and Bonek. It is reproduced with modification from the following paper:
    `Marcos Lopez de Prado “Estimation of Theory-Implied Correlation Matrices”, (2019).
    <https://papers.ssrn.com/abstract_id=3484152>`_.
    """

    def __init__(self):
        """
        Initialize
        """


        pass

    def tic_correlation(self, tree_struct, corr_matrix, tn_relation, kde_bwidth=0.01):
        """
        Calculates the Theory-Implied Correlation (TIC) matrix.

        Includes three steps.

        In the first step, the theoretical tree graph structure of the assets is fit on the evidence
        presented by the empirical correlation matrix.

        The result of the first step is a binary tree (dendrogram) that sequentially clusters two elements
        together, while measuring how closely together the two elements are, until all elements are
        subsumed within the same cluster.

        In the second step, a correlation matrix is derived from the linkage object.

        Each cluster in the global linkage object is decomposed into two elements,
        which can be either atoms or other clusters. Then the off-diagonal correlation between two
        elements is calculated based on the distances between them.

        In the third step, the correlation matrix is de-noised.

        This is done by fitting the Marcenko-Pastur distribution to the eigenvalues of the matrix, calculating the
        maximum theoretical eigenvalue as a threshold and shrinking the eigenvalues higher than a set threshold.
        This algorithm is implemented in the RiskEstimators class.

        :param tree_struct: (pd.dataframe) The tree graph that represents the structure of the assets
        :param corr_matrix: (pd.dataframe) The empirical correlation matrix of the assets
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    correlation matrix
        :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE for de-noising the correlation matrix
                                   (0.01 by default)
        :return: (np.array) Theory-Implied Correlation matrix
        """

        pass

    @staticmethod
    def corr_dist(corr0, corr1):
        """
        Calculates the correlation matrix distance proposed by Herdin and Bonek.

        The distance obtained measures the orthogonality between the considered
        correlation matrices. If the matrices are equal up to a scaling factor,
        the distance becomes zero and one if they are different to a maximum
        extent.

        This can be used to measure to which extent the TIC matrix has blended
        theory-implied views (tree structure of the elements) with empirical
        evidence (correlation matrix).

        :param corr0: (pd.dataframe) First correlation matrix
        :param corr1: (pd.dataframe) Second correlation matrix
        :return: (float) Correlation matrix distance
        """

        pass

    def _get_linkage_corr(self, tree_struct, corr_matrix):
        """
        Fits the theoretical tree graph structure of the assets in a portfolio on the evidence
        presented by the empirical correlation matrix.

        The result is a binary tree (dendrogram) that sequentially clusters two elements
        together, while measuring how closely together the two elements are, until all elements are
        subsumed within the same cluster.

        This is the first step of the TIC algorithm.

        :param tree_struct: (pd.dataframe) The tree graph that represents the structure of the assets
        :param corr_matrix: (pd.dataframe) The empirical correlation matrix of the assets
        :return: (np.array) Linkage object that characterizes the dendrogram
        """

        pass

    @staticmethod
    def _link_clusters(global_linkage, local_linkage, global_elements, grouped_elements):
        """
        Transforms linkage object from local local_linkage (based on dist1) into global global_linkage (based on dist0)

        Consists of changes of names for the elements in clusters and change of the number of
        basic elements (atoms) contained inside a cluster. This is done to take into account the
        already existing links.

        :param global_linkage: (np.array) Global linkage object (previous links)
        :param local_linkage: (np.array) Local linkage object (containing grouped elements and not global ones)
        :param global_elements: (list) List of names for all elements (global)
        :param grouped_elements: (list) List of grouped elements (local)
        :return: (np.array) Local linkage object changed to global one
        """

        pass

    @staticmethod
    def _update_dist(distance_matrix, global_linkage, local_linkage_tr, global_elements, criterion=None):
        """
        Updates the general distance matrix to take the new clusters into account

        Replaces the elements added to the new clusters with these clusters as elements.
        Requires the recalculation of the distance matrix to determine the distance from
        new clusters to other elements.

        A criterion function may be given for calculation of the new distances from a new cluster to other
        elements based on the distances of elements included in a cluster. The default method is the weighted
        average of distances based on the number of atoms in each of the two elements.

        :param distance_matrix: (pd.dataframe) Previous distance matrix
        :param global_linkage: (np.array) Global linkage object that includes new clusters
        :param local_linkage_tr: (np.array) Local linkage object transformed (global names of elements and atoms count)
        :param global_elements: (list) Global list with names of all elements
        :param criterion: (function) Function to apply to a dataframe of distances to adjust them
        :return: (np.array) Updated distance matrix
        """

        pass

    @staticmethod
    def _get_atoms(linkage, element):
        """
        Getting the atoms included in an element from a linkage object

        Atoms are the basic assets in a portfolio and not clusters.

        :param linkage: (np.array) Global linkage object
        :param element: (int) Element id to get atoms from
        :return: (list) Set of atoms
        """

        pass

    def _link2corr(self, linkage, element_index):
        """
        Derives a correlation matrix from the linkage object.

        Each cluster in the global linkage object is decomposed into two elements,
        which can be either atoms or other clusters. Then the off-diagonal correlation between two
        elements are calculated based on the distances between them.

        This is the second step of the TIC algorithm.

        :param linkage: (np.array) Global linkage object
        :param element_index: (pd.index) Names of elements used to calculate the linkage object
        :return: (pd.dataframe) Correlation matrix associated with linkage object
        """

        pass
