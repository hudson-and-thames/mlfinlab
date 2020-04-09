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
        Fits the theoretical tree graph structure on the evidence presented by the empirical
        correlation matrix. The result is a binary tree that sequentially clusters two items
        together, while measuring how closely together the two items are, until all items are
        subsumed within the same cluster.

        The first step of the TIC algorithm.

        :param tree: (list) The tree graph that represents the economic theory
        :param corr: (list) The empirical correlation matrix
        :return: (list) Linkage object that characterizes the dendrogram
        """

        # Adding top level to the tree if it's missing
        if len(np.unique(tree.iloc[:, -1])) > 1:
            tree['All'] = 0

        # Creating a matrix with link arrays.
        # Information in each of the four elements  in the link array is:
        # (1) and (2) - ids of two items clustered together
        # (3) - distance between those items
        # (4) - number of original variables in this cluster
        lnk0 = np.empty(shape=(0, 4))

        # List of lists of two consecutive tree columns
        lvls = [[tree.columns[i-1], tree.columns[i]] for i in range(1, tree.shape[1])]

        # Calculating distance matrix from the correlation matrix
        dist0 = ((1 - corr) / 2)**(1/2)

        # Mapping lnk0 to dist0

        # Getting a list of all element names
        items0 = dist0.index.tolist()

        # Iterating through levels
        for cols in lvls:

            # Groups in a tree
            # Taking two consecutive levels of the tree
            # Removing duplicates from the higher level of the tree
            # Setting the index of the elements in the higher level
            # Grouping by elements in the lower level
            grps = tree[cols].drop_duplicates(cols[0]).set_index(cols[0]).groupby(cols[1])

            # Iterating through two layers - taking the category and the items inside
            for cat, items1 in grps:
                # Getting index of items inside the category
                items1 = items1.index.tolist()

                # If single item then rename
                if len(items1) == 1:
                    # Changing the name of the element in the initial list to the name of the category
                    items0[items0.index(items1[0])] = cat
                    # Changing the name of the element also in the correlation matrix rows and columns
                    dist0 = dist0.rename({items1[0]: cat}, axis=0)
                    dist0 = dist0.rename({items1[0]: cat}, axis=1)
                    continue

                # Taking the distance of the element (?)
                dist1 = dist0.loc[items1, items1]

                # Transforming distance matrix to distance vector
                # Check for matrix symmetry is made
                dist_vec = ssd.squareform(dist1, force='tovector', checks=(not np.allclose(dist1, dist1.T)))

                # Doing hierarchical clustering of the distance vector
                lnk1 = sch.linkage(dist_vec, optimal_ordering=True)

                # Creating a new link array (?)
                lnk_ = self.link_clusters(lnk0, lnk1, items0, items1)

                # Adding new link array to the existing matrix of links
                lnk0 = np.append(lnk0, lnk_, axis=0)

                # As we've created more clusters, their names are added to the global list
                items0 += range(len(items0), len(items0)+len(lnk_))

                # Updating the general distance matrix to take the new cluster in to account
                dist0 = self.update_dist(dist0, lnk0, lnk_, items0)

                # Renaming the last cluster for next level
                items0[-1] = cat

                # Changing the name of the cluster in the distance matrix
                dist0.columns = dist0.columns[:-1].tolist() + [cat]
                dist0.index = dist0.columns

            # Changing from link arrays to link tuples
            lnk0 = np.array(map(tuple, lnk0), dtype=[('i0', int), ('i1', int), ('dist', float), ('num', int)])

        return lnk0

    @staticmethod
    def link_clusters(lnk0, lnk1, items0, items1):
        """
        Creating a new link array.

        Transforming partial link1 (based on dist1) into global link0 (based on dist0)

        :param lnk0: Current list of linkages
        :param lnk1: Hierarchical clustering of the distance vector encoded as a linkage matrix
        :param items0: List of all elements names
        :param items1: Index of items inside the category
        :return: New link array
        """

        # Counting the number of atoms - basic elements  and not clusters
        num_atoms = len(items0)-lnk0.shape[0]

        # Copying the partial link
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
        Updates the general distance matrix to take the new cluster in to account

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
