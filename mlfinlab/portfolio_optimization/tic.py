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

        :param tree: (list) The tree graph
        :param corr: (list) The empirical correlation matrix
        :return: (list) Linkage object.
        """

        # Adding top level
        if len(np.unique(tree.iloc[:, -1])) > 1:
            tree['All'] = 0

        # Creating link matrix
        lnk0 = np.empty(shape=(0, 4))

        # Generator for levels
        lvls = [[tree.columns[i-1], tree.columns[i]] for i in range(1, tree.shape[1])]

        # Calculating distance matrix
        dist0 = ((1 - corr) / 2)**(1/2)

        # Linking lnk0 to dist0
        items0 = dist0.index.tolist()

        # Iterating through levels
        for cols in lvls:

            # Groups in a tree
            grps = tree[cols].drop_duplicates(cols[0]).set_index(cols[0]).groupby(cols[1])
            for cat, items1 in grps:
                # Getting index of items
                items1 = items1.index.tolist()
                if len(items1) == 1: # If single item then rename
                    items0[items0.index(items1[0])] = cat
                    dist0 = dist0.rename({items1[0]: cat}, axis=0)
                    dist0 = dist0.rename({items1[0]: cat}, axis=1)
                    continue

                dist1 = dist0.loc[items1, items1]

                # cluster that cat
                lnk1 = sch.linkage(ssd.squareform(dist1, force='tovector', checks=(not np.allclose(dist1, dist1.T))),
                                   optimal_ordering=True)

                lnk_ = self.link_clusters(lnk0, lnk1, items0, items1)
                lnk0 = np.append(lnk0, lnk_, axis=0)
                items0 += range(len(items0), len(items0)+len(lnk_))
                dist0 = self.update_dist(dist0, lnk0, lnk_, items0)

                # Rename last cluster for next level
                items0[-1] = cat
                dist0.columns = dist0.columns[:-1].tolist() + [cat]
                dist0.index = dist0.columns
            lnk0 = np.array(map(tuple, lnk0), dtype=[('i0', int), ('i1', int), ('dist', float), ('num', int)])

        return lnk0

    @staticmethod
    def link_clusters(lnk0, lnk1, items0, items1):
        """


        :param lnk0:
        :param lnk1:
        :param items0:
        :param items1:
        :return:
        """

        # transform partial link1 (based on dist1) into global link0 (based on dist0)
        num_atoms = len(items0)-lnk0.shape[0]
        lnk_ = lnk1.copy()
        for i in range(lnk_.shape[0]):
            el_i3 = 0
            for j in range(2):
                if lnk_[i, j] < len(items1):
                    lnk_[i, j] = items0.index(items1[int(lnk_[i, j])])
                else:
                    lnk_[i, j] += -len(items1) + len(items0)

                # update number of items
                if lnk_[i, j] < num_atoms:
                    el_i3 += 1
                else:
                    if lnk_[i, j] - num_atoms < lnk0.shape[0]:
                        el_i3 += lnk0[int(lnk_[i, j]) - num_atoms, 3]
                    else:
                        el_i3 += lnk_[int(lnk_[i, j]) - len(items0), 3]
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
