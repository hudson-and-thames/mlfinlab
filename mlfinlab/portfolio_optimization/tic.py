# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators

class TIC:
    """
    This class implements the Theory-Implied Correlation (TIC) algorithm. It is reproduced with
    modification from the following paper: `Marcos Lopez de Prado “Estimation of Theory-Implied Correlation Matrices”,
    (2019). <https://papers.ssrn.com/abstract_id=3484152>`_.
    """

    def __init__(self):
        """
        Initialize
        """

        return

    def get_linkage_corr(self, tree, corr):
        """
        Fits the theoretical tree graph structure of the assets in a portfolio on the evidence
        presented by the empirical correlation matrix.

        The result is a binary tree (dendrogram) that sequentially clusters two elements
        together, while measuring how closely together the two elements are, until all elements are
        subsumed within the same cluster.

        This is the first step of the TIC algorithm.

        :param tree: (list) The tree graph that represents the structure of the assets
        :param corr: (list) The empirical correlation matrix of the assets
        :return: (list) Linkage object that characterizes the dendrogram
        """

        # If the top level of the tree contains multiple elements, creating a level with just one element (tree root)
        if len(np.unique(tree.iloc[:, -1])) > 1:
            tree['All'] = 0

        # Creating a linkage object (matrix with link elements).
        # Each row represents a cluster and consists of the following columns:
        # (1) and (2) - number ids of two elements clustered together
        # (3) - distance between those elements
        # (4) - number of original variables in this cluster
        # Items in a cluster can be single elements or other clusters
        lnk0 = np.empty(shape=(0, 4))

        # List with elements containing two consecutive tree levels
        lvls = [[tree.columns[i-1], tree.columns[i]] for i in range(1, tree.shape[1])]

        # Calculating distance matrix from the empirical correlation matrix
        dist0 = ((1 - corr) / 2)**(1/2)

        # Getting a list of names of assets asset
        items0 = dist0.index.tolist()

        # Iterating through levels of the tree
        for cols in lvls:

            # Taking two consecutive levels of the tree
            # Removing duplicates from the lower level of the tree
            # Setting the obtained unique elements from the lower level as index
            # Grouping by elements in the higher level
            grps = tree[cols].drop_duplicates(cols[0]).set_index(cols[0]).groupby(cols[1])

            # Iterating through the obtained two levels of a tree
            for cat, items1 in grps:
                # cat contains the higher element
                # items1 contain the elements from lower level, grouped under the higher level element

                # Getting the names of the grouped elements
                items1 = items1.index.tolist()

                # If only one element grouped by the element from the higher level
                if len(items1) == 1:
                    # Changing the name of this element to the name of element from the higher level
                    # As this element is now representing the higher level

                    # Changing the name in the list of names
                    items0[items0.index(items1[0])] = cat

                    # Changing the name also in the correlation matrix rows and columns
                    dist0 = dist0.rename({items1[0]: cat}, axis=0)
                    dist0 = dist0.rename({items1[0]: cat}, axis=1)

                    continue

                # Taking the part of the distance matrix containing the grouped elements in the lower level
                dist1 = dist0.loc[items1, items1]

                # Transforming distance matrix to distance vector
                # Check for matrix symmetry is made - checking that the matrix given is a distance matrix
                dist_vec = ssd.squareform(dist1, force='tovector', checks=(not np.allclose(dist1, dist1.T)))

                # Doing hierarchical clustering of the distance vector. Result is a linkage object
                # Here we have created new clusters based on the grouped elements
                lnk1 = sch.linkage(dist_vec, optimal_ordering=True)

                # Transforming the linkage object from local (only containing the grouped elements)
                # to global (containing all elements)
                lnk_ = self.link_clusters(lnk0, lnk1, items0, items1)

                # Adding new link elements to the general linage object
                lnk0 = np.append(lnk0, lnk_, axis=0)

                # As more clusters were created, their names are added to the global list of elements
                items0 += range(len(items0), len(items0)+len(lnk_))

                # Updating the general distance matrix to take the new clusters into account
                # Now the grouped elements in the distance matrix will be replaced with new clusters as elements.
                dist0 = self.update_dist(dist0, lnk0, lnk_, items0)

                # The last added cluster is representing the element from the higher level
                # So we're changing the name of that cluster to the name of the higher level element
                items0[-1] = cat

                # Changing the name of the cluster also in the distance matrix
                dist0.columns = dist0.columns[:-1].tolist() + [cat]
                dist0.index = dist0.columns

        # Changing the linkage object from array of arrays to array of tuples with named fields
        lnk0 = np.array([*map(tuple, lnk0)], dtype=[('i0', int), ('i1', int), ('dist', float), ('num', int)])

        return lnk0

    @staticmethod
    def link_clusters(lnk0, lnk1, items0, items1):
        """
        Transforms linkage object from local link1 (based on dist1) into global link0 (based on dist0)

        Consists of changes of names for the elements in clusters and change of the number of
        basic elements (atoms) contained inside a cluster. This is done to take into account the
        already existing links.

        :param lnk0: Global linkage object (previous links)
        :param lnk1: Local linkage object (containing grouped elements and not global ones)
        :param items0: List of names for all elements (global)
        :param items1: List of grouped elements (local)
        :return: Local linkage object changed to global one
        """

        # Counting the number of atoms - basic elements and not clusters
        # It's the total number of elements minus the number of links (each link represents 1 cluster)
        num_atoms = len(items0) - lnk0.shape[0]

        # Making a copy of a local linkage object
        lnk_ = lnk1.copy()

        # Iterating through links in the local linkage object
        for i in range(lnk_.shape[0]):
            # Counting the number of atoms in the cluster (represented by this link)
            el_i3 = 0

            # Iterating through the two elements contained in a cluster (represented by this link)
            for j in range(2):
                # Changing the names in links to global ones

                if lnk_[i, j] < len(items1):  # If it's the element from the grouped ones
                    # Then replacing its local name with the actual name from the list of all elements' names
                    lnk_[i, j] = items0.index(items1[int(lnk_[i, j])])

                else:  # Otherwise it's a new cluster
                    # Then giving it a new name, taking into account the previously named clusters
                    # The names of the clusters are sequential numbers
                    lnk_[i, j] += -len(items1) + len(items0)

                # Updating the number of atoms in a cluster (represented by this link)

                if lnk_[i, j] < num_atoms:  # If the added element is an atom
                    # Then add one to the counter of atoms inside
                    el_i3 += 1

                else:  # If the element added is a cluster
                    # If the added element is a previously created cluster
                    if lnk_[i, j] - num_atoms < lnk0.shape[0]:
                        # Adding to counter the number of atoms from the global linkage object
                        el_i3 += lnk0[int(lnk_[i, j]) - num_atoms, 3]

                    else:  # If the added element is a newly created cluster
                        # Adding to the counter the number of atoms from the local linkage object
                        el_i3 += lnk_[int(lnk_[i, j]) - len(items0), 3]

            # Setting the number of atoms in the cluster to the calculated counter
            lnk_[i, 3] = el_i3

        return lnk_

    @staticmethod
    def update_dist(dist0, lnk0, lnk_, items0, criterion=None):
        """
        Updates the general distance matrix to take the new clusters into account

        Replaces the elements added to the new clusters with these clusters as elements.
        Requires the recalculation of the distance matrix to determine the distance from
        new clusters to other elements.

        :param dist0: Previous distance matrix
        :param lnk0: Global linkage object that includes new clusters
        :param lnk_: Local linkage object updated to global names of elements and number of contained atoms
        :param items0: Global list with names of all elements
        :param criterion: Function to apply to a dataframe of distances to adjust them
        :return: Updated distance matrix
        """

        # Counting the number of atoms - basic elements  and not clusters
        num_atoms = len(items0) - lnk0.shape[0]

        # Getting the list with names of new items
        new_items = items0[-lnk_.shape[0]:]

        # Iterating through elements in the local linkage object
        for i in range(lnk_.shape[0]):
            # Getting the names of two elements clustered together
            el_i0, el_i1 = items0[int(lnk_[i, 0])], items0[int(lnk_[i, 1])]

            # If no criterion function given to determine new distances then the weighted average
            # based on the number of atoms in each of the two elements is used
            if criterion is None:

                if lnk_[i, 0] < num_atoms:  # If the first element is an atom
                    # Weight of the element is 1
                    el_w0 = 1

                else:  # If the first element is a cluster
                    # Weight is set to the number of atoms in a cluster
                    el_w0 = lnk0[int(lnk_[i, 0]) - num_atoms, 3]

                if lnk_[i, 1] < num_atoms:  # If the second element is an atom
                    # Weight of the element is 1
                    el_w1 = 1

                else:  # If the second element is a cluster
                    # Weight is set to the number of atoms in a cluster
                    el_w1 = lnk0[int(lnk_[i, 1]) - num_atoms, 3]

                # Calculating new distance as the average weighted distance
                # where the weight is the number of atoms in an element
                dist1 = (dist0[el_i0] * el_w0 + dist0[el_i1] * el_w1) / (el_w0 + el_w1)

            # If criterion function is given, the new distance is calculated using it
            else:
                # New distance
                dist1 = criterion(dist0[[el_i0, el_i1]], axis=1)

            # Adding column with new cluster to the distance matrix
            dist0[new_items[i]] = dist1

            # Adding row with new cluster to the distance matrix
            dist0.loc[new_items[i]] = dist1

            # Setting the main diagonal value for the new cluster to 0 (distance of the element to itself is zero)
            dist0.loc[new_items[i], new_items[i]] = 0

            # And deleting the two elements that were combined in the new cluster
            dist0 = dist0.drop([el_i0, el_i1], axis=0)
            dist0 = dist0.drop([el_i0, el_i1], axis=1)

        return dist0

    @staticmethod
    def get_atoms(lnk, item):
        """
        Getting the atoms included in an element from a linkage object

        Atoms are the basic assets in a portfolio and not clusters.

        :param lnk: Global linkage object
        :param item: Element to get atoms from
        :return: Set of atoms
        """

        # A list of elements to unpack
        # Now includes only one given element, but will be appended with the unpacked elements
        anc = [item]

        # Iterating (until there are elements to unpack)
        while True:
            # The maximum item from the list (as the clusters have higher numbers in comparison to atoms)
            item_ = max(anc)

            # If it's a cluster and not an atom
            if item_ > lnk.shape[0]:
                # Delete this cluster
                anc.remove(item_)

                # Unpack the elements in a cluster and add them to a list of elements to unpack
                anc.append(lnk['i0'][item_ - lnk.shape[0] - 1])
                anc.append(lnk['i1'][item_ - lnk.shape[0] - 1])

            else:  # If all the elements left in the list are atoms, we're done
                break

        # The resulting list contains only atoms
        return anc

    def link2corr(self, lnk, lbls):
        """
        Derives a correlation matrix from the linkage object.

        Each cluster in the global linkage object is decomposed to two elements,
        which can be either atoms or other clusters. Then the off -diagonal correlation between two
        elements are calculated based on the distances between them.

        This is the second step of the TIC algorithm.

        :param lnk: Global linkage object
        :param lbls: Names of elements used to calculate the linkage object
        :return: Correlation matrix associated with linkage object
        """

        # Creating a base for new correlation matrix with ones on the main diagonal
        corr = pd.DataFrame(np.eye(lnk.shape[0]+1), index=lbls, columns=lbls, dtype=float)

        # Iterating through links in the linkage object
        for i in range(lnk.shape[0]):
            # Getting the atoms contained in the first element from the link
            el_x = self.get_atoms(lnk, lnk['i0'][i])

            # Getting the atoms contained in the second element from the link
            el_y = self.get_atoms(lnk, lnk['i1'][i])

            # Calculating the odd-diagonal values of the correlation matrix
            corr.loc[lbls[el_x], lbls[el_y]] = 1 - 2 * lnk['dist'][i]**2

            # And the symmetrical values
            corr.loc[lbls[el_y], lbls[el_x]] = 1 - 2 * lnk['dist'][i]**2

        return corr

    def tic_correlation(self, tree, corr, tn_relation, kde_bwidth):
        """
        Calculates the Theory-Implied Correlation (TIC) matrix.

        Includes two steps.

        On the first step, the theoretical tree graph structure of the assets is fit on the evidence
        presented by the empirical correlation matrix.

        The result of the first step is a binary tree (dendrogram) that sequentially clusters two elements
        together, while measuring how closely together the two elements are, until all elements are
        subsumed within the same cluster.

        On the second step, a correlation matrix is derived from the linkage object.

        Each cluster in the global linkage object is decomposed to two elements,
        which can be either atoms or other clusters. Then the off -diagonal correlation between two
        elements are calculated based on the distances between them.

        :param tree: (list) The tree graph that represents the structure of the assets
        :param corr: (list) The empirical correlation matrix of the assets
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    correlation matrix
        :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE for de-noising the correlation matrix
        :return: (list) Theory-Implies Correlation matrix
        """

        # Getting the linkage object that characterizes the dendrogram
        lnk0 = self.get_linkage_corr(tree, corr)

        # Calculating the correlation matrix from the dendrogram
        corr0 = self.link2corr(lnk0, corr.index)

        # Class with function for de-noising the correlation matrix
        risk_estim = RiskEstimators()

        # De-noising the obtained Theory-Implies Correlation matrix
        corr1 = risk_estim.denoise_covariance(corr0, tn_relation=tn_relation, kde_bwidth=kde_bwidth)

        return corr1

    @staticmethod
    def corr_dist(corr0, corr1):
        """
        Calculates correlation matrix distance proposed by Herdin and Bonek.

        The distance obtained measures the orthogonality between the considered
        correlation matrices. If the matrices are equal up to a scaling factor,
        the distance becomes zero and one if they are different to a maximum
        extent.

        This can be used to measure to which extent the TIC matrix has blended
        theory-implied views (tree structure of the elements) with empirical
        evidence (correlation matrix).

        :param corr0: First correlation matrix
        :param corr1: Second correlation matrix
        :return: Correlation matrix distance
        """

        # Trace of the product of correlation matrices
        num = np.trace(np.dot(corr0, corr1))

        # Frobenius norm of the first correlation matrix
        den = np.linalg.norm(corr0, ord='fro')

        # Frobenius norm of the second correlation matrix
        den *= np.linalg.norm(corr1, ord='fro')

        # Distance calculation
        cmd = 1 - num / den

        return cmd
