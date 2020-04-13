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

        The result is a binary tree (dendrogram) that sequentially clusters two items
        together, while measuring how closely together the two items are, until all items are
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
        # (1) and (2) - number ids of two items clustered together
        # (3) - distance between those items
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
        Transforming linkage object from local link1 (based on dist1) into global link0 (based on dist0)

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

        # Iterating through elements in the partial link
        for i in range(lnk_.shape[0]):
            # Setting the number of the atom elements in the cluster
            el_i3 = 0

            # Iterating through the second index of elements
            for j in range(2):
                if lnk_[i, j] < len(items1):  # If the element is smaller than the num of items in the category
                    # Then replacing it with name from all elements
                    lnk_[i, j] = items0.index(items1[int(lnk_[i, j])])
                else:
                    # Otherwise it's global element number minus the local element number
                    lnk_[i, j] += -len(items1) + len(items0)

                # Update number of items

                # If the added element is an atom
                if lnk_[i, j] < num_atoms:
                    # Then add one to the counter
                    el_i3 += 1
                else:  # If the element added is a cluster
                    # If the added element is in the list of added clusters previously
                    if lnk_[i, j] - num_atoms < lnk0.shape[0]:
                        # Adding to counter the number of atoms from the global link matrix
                        el_i3 += lnk0[int(lnk_[i, j]) - num_atoms, 3]
                    else:  # If the added element is the newly added cluster
                        # Adding to the counter the number of atims from the local link matrix
                        el_i3 += lnk_[int(lnk_[i, j]) - len(items0), 3]

            # Setting the number of atoms in the cluster to the according value.
            lnk_[i, 3] = el_i3

        return lnk_

    @staticmethod
    def update_dist(dist0, lnk0, lnk_, items0, criterion=None):
        """
        Updates the general distance matrix to take the new cluster into account

        Expands dist0 to incorporate newly created clusters

        :param dist0: Distance matrix with all elements
        :param lnk0: Matrix of links
        :param lnk_: New link array
        :param items0: List of all element names
        :param criterion: function of linkage criterion
        :return: Updated distance matrix
        """

        # Counting the number of atoms - basic elements  and not clusters
        num_atoms = len(items0) - lnk0.shape[0]

        # Getting the list with names of new items
        new_items = items0[-lnk_.shape[0]:]

        # Iterating through elements in the partial link
        for i in range(lnk_.shape[0]):
            # Ids of two items clustered together
            el_i0, el_i1 = items0[int(lnk_[i, 0])], items0[int(lnk_[i, 1])]

            # If no criterion function given
            if criterion is None:

                # If the first element is an atom
                if lnk_[i, 0] < num_atoms:
                    # Weight is set to 1
                    el_w0 = 1.
                else:  # If the first element is a cluster
                    # Weight is set as number of elements in a cluster
                    el_w0 = lnk0[int(lnk_[i, 0]) - num_atoms, 3]

                # If the second element is an atom
                if lnk_[i, 1] < num_atoms:
                    # Weight is set to 1
                    el_w1 = 1.
                else:  # If the second element is a cluster
                    # Weight is set as number of elements in a cluster
                    el_w1 = lnk0[int(lnk_[i, 1]) - num_atoms, 3]

                # Calculating new distance as the average weighted distances
                # where the weight is number of atoms in an element
                dist1 = (dist0[el_i0] * el_w0 + dist0[el_i1] * el_w1) / (el_w0 + el_w1)

            # If criterion function is given, the new distance is calculated using it.
            else:
                # New distance
                dist1 = criterion(dist0[[el_i0, el_i1]], axis=1)

            # Adding column with new element
            dist0[new_items[i]] = dist1

            # Adding row with new element
            dist0.loc[new_items[i]] = dist1

            # Setting the main diagonal value for the new element to 0
            dist0.loc[new_items[i], new_items[i]] = 0

            # And deleting the two elements that were combined in the new element
            dist0 = dist0.drop([el_i0, el_i1], axis=0)
            dist0 = dist0.drop([el_i0, el_i1], axis=1)

        return dist0

    @staticmethod
    def get_atoms(lnk, item):
        """
        Getting all atoms included in an item

        :param lnk: Matrix of links
        :param item: Item to get atoms from
        :return: Set of atoms
        """

        # A list of items to unpack
        # Now includes only one item
        anc = [item]

        # Iterating
        while True:
            # The maximum item from the list
            item_ = max(anc)

            # If it's a cluster and not an atom
            if item_ > lnk.shape[0]:
                # Delete this cluster
                anc.remove(item_)

                # Adding the elements of the cluster to list of items to unpack
                anc.append(lnk['i0'][item_ - lnk.shape[0] - 1])
                anc.append(lnk['i1'][item_ - lnk.shape[0] - 1])

            else:  # If all the items left in the list are atoms, we're done
                break

        return anc

    def link2corr(self, lnk, lbls):
        """
        Derives a correlation matrix from the linkage object. Each cluster
        is decomposed to two elements, which can be either atoms or other
        clusters.

        The second step of the TIC algorithm.

        :param lnk: Matrix of links
        :param lbls: Labels
        :return: Correlation matrix associated with linkage matrix
        """

        # Creating a base for correlation matrix with ones on the main diagonal
        corr = pd.DataFrame(np.eye(lnk.shape[0]+1), index=lbls, columns=lbls, dtype=float)

        # Iterating through links
        for i in range(lnk.shape[0]):
            # Getting the first element in a link
            el_x = self.get_atoms(lnk, lnk['i0'][i])

            # Getting the second element in a link
            el_y = self.get_atoms(lnk, lnk['i1'][i])

            # Calculating the odd-diagonal values of the correlation matrix
            corr.loc[lbls[el_x], lbls[el_y]] = 1 - 2 * lnk['dist'][i]**2

            # And the symmetrical values
            corr.loc[lbls[el_y], lbls[el_x]] = 1 - 2 * lnk['dist'][i]**2

        return corr

    def tic_correlation(self, tree, corr, tn_relation, kde_bwidth):
        """
        Calculating the Theory-Implies Correlation (TIC) matrix.

        :param tree: (list) The tree graph that represents the economic theory
        :param corr: (list) The empirical correlation matrix
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    correlation matrix.
        :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE
        :return: (list) TIC matrix
        """

        # Getting the linkage object that characterizes the dendrogram
        lnk0 = self.get_linkage_corr(tree, corr)

        # Calculating the correlation matrix from the dendrogram
        corr0 = self.link2corr(lnk0, corr.index)

        # Class with function for de-noising correlation matrix
        risk_estim = RiskEstimators()

        # De-noising the obtained correlation matrix
        corr1 = risk_estim.denoise_covariance(corr0, tn_relation=tn_relation, kde_bwidth=kde_bwidth)

        return corr1

    @staticmethod
    def corr_dist(corr0, corr1):
        """
        Calculates correlation matrix distance proposed by Herdin and Bonek.

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
