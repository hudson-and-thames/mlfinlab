# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators

class TIC:
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

        return

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

        # If the top level of the tree contains multiple elements, creating a level with just one element (tree root)
        if len(np.unique(tree_struct.iloc[:, -1])) > 1:
            tree_struct['All'] = 0

        # Creating a linkage object (matrix with link elements).
        # Each row represents a cluster and consists of the following columns:
        # (1) and (2) - number ids of two elements clustered together
        # (3) - the distance between those elements
        # (4) - number of original variables in this cluster
        # Items in a cluster can be single elements or other clusters
        global_linkage = np.empty(shape=(0, 4))

        # List with elements containing two consecutive tree levels
        tree_levels = [[tree_struct.columns[i-1], tree_struct.columns[i]] for i in range(1, tree_struct.shape[1])]

        # Calculating the distance matrix from the empirical correlation matrix
        distance_matrix = ((1 - corr_matrix) / 2)**(1/2)

        # Getting a list of names of elements
        global_elements = distance_matrix.index.tolist()

        # Iterating through levels of the tree
        for level in tree_levels:

            # Taking two consecutive levels of the tree
            # Removing duplicates from the lower level of the tree
            # Setting the obtained unique elements from the lower level as an index
            # Grouping by elements in the higher level
            grouped_level = tree_struct[level].drop_duplicates(level[0]).set_index(level[0]).groupby(level[1])

            # Iterating through the obtained two levels of a tree
            for high_element, grouped_elements in grouped_level:
                # high_element contains the higher element
                # grouped_elements contain the elements from the lower level, grouped under the higher-level element

                # Getting the names of the grouped elements
                grouped_elements = grouped_elements.index.tolist()

                # If only one element grouped by the element from the higher level
                if len(grouped_elements) == 1:
                    # Changing the name of this element to the name of the element from the higher level
                    # As this element is now representing the higher level

                    # Changing the name in the list of names
                    global_elements[global_elements.index(grouped_elements[0])] = high_element

                    # Changing the name also in the correlation matrix rows and columns
                    distance_matrix = distance_matrix.rename({grouped_elements[0]: high_element}, axis=0)
                    distance_matrix = distance_matrix.rename({grouped_elements[0]: high_element}, axis=1)

                    continue

                # Taking the part of the distance matrix containing the grouped elements in the lower level
                local_distance = distance_matrix.loc[grouped_elements, grouped_elements]

                # Transforming distance matrix to distance vector
                # Check for matrix symmetry is made - checking that the matrix given is a distance matrix
                distance_vec = ssd.squareform(local_distance, force='tovector',
                                              checks=(not np.allclose(local_distance, local_distance.T)))

                # Doing hierarchical clustering of the distance vector. Result is a linkage object
                # Here we have created new clusters based on the grouped elements
                local_linkage = sch.linkage(distance_vec, optimal_ordering=True)

                # Transforming the linkage object from local (only containing the grouped elements)
                # to global (containing all elements) form
                local_linkage_transformed = self._link_clusters(global_linkage, local_linkage, global_elements,
                                                                grouped_elements)

                # Adding new link elements to the general linage object
                global_linkage = np.append(global_linkage, local_linkage_transformed, axis=0)

                # As more clusters were created, their names are added to the global list of elements
                global_elements += range(len(global_elements), len(global_elements) + len(local_linkage_transformed))

                # Updating the general distance matrix to take the new clusters into account
                # Now the grouped elements in the distance matrix will be replaced with new clusters as elements.
                distance_matrix = self._update_dist(distance_matrix, global_linkage, local_linkage_transformed,
                                                    global_elements)

                # The last added cluster is representing the element from the higher level
                # So we're changing the name of that cluster to the name of the higher level element
                global_elements[-1] = high_element

                # Changing the name of the cluster also in the distance matrix
                distance_matrix.columns = distance_matrix.columns[:-1].tolist() + [high_element]
                distance_matrix.index = distance_matrix.columns

        # Changing the linkage object from an array of arrays to array of tuples with named fields
        global_linkage = np.array([*map(tuple, global_linkage)],
                                  dtype=[('i0', int), ('i1', int), ('dist', float), ('num', int)])

        return global_linkage

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

        # Counting the number of atoms - basic elements and not clusters
        # It's the total number of elements minus the number of links (each link represents 1 cluster)
        num_atoms = len(global_elements) - global_linkage.shape[0]

        # Making a copy of a local linkage object
        local_linkage_tr = local_linkage.copy()

        # Iterating through links in the local linkage object
        for link in range(local_linkage_tr.shape[0]):
            # Counting the number of atoms in the cluster (represented by this link)
            atom_counter = 0

            # Iterating through the two elements contained in a cluster (represented by this link)
            for j in range(2):
                # Changing the names in links to global ones

                if local_linkage_tr[link, j] < len(grouped_elements):  # If it's the element from the grouped ones
                    # Then replacing its local name with the actual name from the list of all elements' names
                    local_linkage_tr[link, j] = global_elements.index(grouped_elements[int(local_linkage_tr[link, j])])

                else:  # Otherwise it's a new cluster
                    # Then giving it a new name, taking into account the previously named clusters
                    # The names of the clusters are sequential numbers
                    local_linkage_tr[link, j] += -len(grouped_elements) + len(global_elements)

                # Updating the number of atoms in a cluster (represented by this link)

                if local_linkage_tr[link, j] < num_atoms:  # If the added element is an atom
                    # Then add one to the counter of atoms inside
                    atom_counter += 1

                else:  # If the element added is a cluster
                    # If the added element is a previously created cluster
                    if local_linkage_tr[link, j] - num_atoms < global_linkage.shape[0]:
                        # Adding to counter the number of atoms from the global linkage object
                        atom_counter += global_linkage[int(local_linkage_tr[link, j]) - num_atoms, 3]

                    else:  # If the added element is a newly created cluster
                        # Adding to the counter the number of atoms from the local linkage object
                        atom_counter += local_linkage_tr[int(local_linkage_tr[link, j]) - len(global_elements), 3]

            # Setting the number of atoms in the cluster to the calculated counter
            local_linkage_tr[link, 3] = atom_counter

        return local_linkage_tr

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

        # Counting the number of atoms - basic elements  and not clusters
        num_atoms = len(global_elements) - global_linkage.shape[0]

        # Getting the list with names of new items
        new_items = global_elements[-local_linkage_tr.shape[0]:]

        # Iterating through elements in the local linkage object
        for i in range(local_linkage_tr.shape[0]):
            # Getting the names of two elements clustered together
            elem_1, elem_2 = global_elements[int(local_linkage_tr[i, 0])], global_elements[int(local_linkage_tr[i, 1])]

            # If no criterion function given to determine new distances then the weighted average
            # based on the number of atoms in each of the two elements is used
            if criterion is None:

                if local_linkage_tr[i, 0] < num_atoms:  # If the first element is an atom
                    # Weight of the element is 1
                    elem_1_weight = 1

                else:  # If the first element is a cluster
                    # Weight is set to the number of atoms in a cluster
                    elem_1_weight = global_linkage[int(local_linkage_tr[i, 0]) - num_atoms, 3]

                if local_linkage_tr[i, 1] < num_atoms:  # If the second element is an atom
                    # Weight of the element is 1
                    elem_2_weight = 1

                else:  # If the second element is a cluster
                    # Weight is set to the number of atoms in a cluster
                    elem_2_weight = global_linkage[int(local_linkage_tr[i, 1]) - num_atoms, 3]

                # Calculating new distance as the average weighted distance
                # where the weight is the number of atoms in an element
                dist_vector = (distance_matrix[elem_1] * elem_1_weight + distance_matrix[elem_2] * elem_2_weight) / \
                              (elem_1_weight + elem_2_weight)

            # If a criterion function is given, the new distance is calculated using it
            else:
                # New distance
                dist_vector = criterion(distance_matrix[[elem_1, elem_2]], axis=1)

            # Adding a column with th new cluster to the distance matrix
            distance_matrix[new_items[i]] = dist_vector

            # Adding row with the new cluster to the distance matrix
            distance_matrix.loc[new_items[i]] = dist_vector

            # Setting the main diagonal value for the new cluster to 0 (the distance of the element to itself is zero)
            distance_matrix.loc[new_items[i], new_items[i]] = 0

            # And deleting the two elements that were combined in the new cluster
            distance_matrix = distance_matrix.drop([elem_1, elem_2], axis=0)
            distance_matrix = distance_matrix.drop([elem_1, elem_2], axis=1)

        return distance_matrix

    @staticmethod
    def _get_atoms(linkage, element):
        """
        Getting the atoms included in an element from a linkage object

        Atoms are the basic assets in a portfolio and not clusters.

        :param linkage: (np.array) Global linkage object
        :param element: (int) Element id to get atoms from
        :return: (list) Set of atoms
        """

        # A list of elements to unpack
        # Now includes only one given element but will be appended with the unpacked elements
        element_list = [element]

        # Iterating (until there are elements to unpack)
        while True:
            # The maximum item from the list (as the clusters have higher numbers in comparison to atoms)
            item_ = max(element_list)

            # If it's a cluster and not an atom
            if item_ > linkage.shape[0]:
                # Delete this cluster
                element_list.remove(item_)

                # Unpack the elements in a cluster and add them to a list of elements to unpack
                element_list.append(linkage['i0'][item_ - linkage.shape[0] - 1])
                element_list.append(linkage['i1'][item_ - linkage.shape[0] - 1])

            else:  # If all the elements left in the list are atoms, we're done
                break

        # The resulting list contains only atoms
        return element_list

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

        # Creating a base for new correlation matrix with ones on the main diagonal
        corr_matrix = pd.DataFrame(np.eye(linkage.shape[0]+1), index=element_index, columns=element_index, dtype=float)

        # Iterating through links in the linkage object
        for link in range(linkage.shape[0]):
            # Getting the atoms contained in the first element from the link
            el_x = self._get_atoms(linkage, linkage['i0'][link])

            # Getting the atoms contained in the second element from the link
            el_y = self._get_atoms(linkage, linkage['i1'][link])

            # Calculating the odd-diagonal values of the correlation matrix
            corr_matrix.loc[element_index[el_x], element_index[el_y]] = 1 - 2 * linkage['dist'][link]**2

            # And the symmetrical values
            corr_matrix.loc[element_index[el_y], element_index[el_x]] = 1 - 2 * linkage['dist'][link]**2

        return corr_matrix

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

        # Getting the linkage object that characterizes the dendrogram
        lnkage_object = self._get_linkage_corr(tree_struct, corr_matrix)

        # Calculating the correlation matrix from the dendrogram
        ti_correlation = self._link2corr(lnkage_object, corr_matrix.index)

        # Class with function for de-noising the correlation matrix
        risk_estim = RiskEstimators()

        # De-noising the obtained Theory-Implies Correlation matrix
        ti_correlation_denoised = risk_estim.denoise_covariance(ti_correlation, tn_relation=tn_relation,
                                                                kde_bwidth=kde_bwidth)

        return ti_correlation_denoised

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

        # Trace of the product of correlation matrices
        prod_trace = np.trace(np.dot(corr0, corr1))

        # Frobenius norm of the first correlation matrix
        frob_product = np.linalg.norm(corr0, ord='fro')

        # Frobenius norm of the second correlation matrix
        frob_product *= np.linalg.norm(corr1, ord='fro')

        # Distance calculation
        distance = 1 - prod_trace / frob_product

        return distance
