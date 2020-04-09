# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch

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


        :param dist0:
        :param lnk0:
        :param lnk_:
        :param items0:
        :param criterion:
        :return:
        """

        # expand dist0 to incorporate newly created clusters
        num_atoms = len(items0) - lnk0.shape[0]
        new_items = items0[-lnk_.shape[0]:]
        for i in range(lnk_.shape[0]):
            el_i0, el_i1 = items0[int(lnk_[i, 0])], items0[int(lnk_[i, 1])]
            if criterion is None:
                if lnk_[i, 0] < num_atoms:
                    el_w0 = 1.
                else:
                    el_w0 = lnk0[int(lnk_[i, 0]) - num_atoms, 3]
                if lnk_[i, 1] < num_atoms:
                    el_w1 = 1.
                else:
                    el_w1 = lnk0[int(lnk_[i, 1]) - num_atoms, 3]
                dist1 = (dist0[el_i0] * el_w0 + dist0[el_i1] * el_w1) / (el_w0 + el_w1)
            else:
                dist1 = criterion(dist0[[el_i0, el_i1]], axis=1)  # linkage criterion
            dist0[new_items[i]] = dist1  # add column
            dist0.loc[new_items[i]] = dist1  # add row
            dist0.loc[new_items[i], new_items[i]] = 0  # main diagonal
            dist0 = dist0.drop([el_i0, el_i1], axis=0)
            dist0 = dist0.drop([el_i0, el_i1], axis=1)

        return dist0

    @staticmethod
    def get_atoms(lnk, item):
        """


        :param lnk:
        :param item:
        :return:
        """

        # get all atoms included in an item
        anc = [item]

        while True:
            item_ = max(anc)
            if item_ > lnk.shape[0]:
                anc.remove(item_)
                anc.append(lnk['i0'][item_ - lnk.shape[0] - 1])
                anc.append(lnk['i1'][item_ - lnk.shape[0] - 1])
            else:
                break
        return anc

    def link2corr(self, lnk, lbls):
        """


        :param lnk:
        :param lbls:
        :return:
        """

        # derive the correl matrix associated with a given linkage matrix
        corr = pd.DataFrame(np.eye(lnk.shape[0]+1), index=lbls, columns=lbls, dtype=float)
        for i in range(lnk.shape[0]):
            el_x = self.get_atoms(lnk, lnk['i0'][i])
            el_y = self.get_atoms(lnk, lnk['i1'][i])
            corr.loc[lbls[el_x], lbls[el_y]] = 1 - 2 * lnk['dist'][i]**2 # off-diagonal values
            corr.loc[lbls[el_y], lbls[el_x]] = 1 - 2 * lnk['dist'][i]**2 # symmetry
        return corr

    @staticmethod
    def corr_dist(corr0, corr1):
        """


        :param corr0:
        :param corr1:
        :return:
        """

        num = np.trace(np.dot(corr0, corr1))
        den = np.linalg.norm(corr0, ord='fro')
        den *= np.linalg.norm(corr1, ord='fro')
        cmd = 1 - num / den
        return cmd
