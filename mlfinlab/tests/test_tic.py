# pylint: disable=protected-access
"""
Tests the Theory-Implied Correlation (TIC) algorithm and the correlation matrix distance function.
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation
from mlfinlab.portfolio_optimization.tic import TIC


class TestTIC(unittest.TestCase):
    """
    Tests different functions of the TIC algorithm class.
    """

    def setUp(self):
        """
        Initialize and load data
        """

        project_path = os.path.dirname(__file__)

        # Loading the price series of ETFs
        price_data_path = project_path + '/test_data/stock_prices.csv'
        self.price_data = pd.read_csv(price_data_path, parse_dates=True, index_col="Date")

        # Transforming series of prices to series of returns
        ret_est = ReturnsEstimation()
        self.returns_data = ret_est.calculate_returns(self.price_data)

        # Loading the classification tree of ETFs
        classification_tree_path = project_path + '/test_data/classification_tree.csv'
        self.classification_tree = pd.read_csv(classification_tree_path)

    def test_get_linkage_corr(self):
        """
        Testing the creation of a linkage object from empirical correlation matrix and tree graph
        """

        tic = TIC()

        # Taking the first 5 ETFs for test purposes
        etf_prices = self.returns_data.iloc[:, :5]
        etf_classification_tree = self.classification_tree.iloc[:5]

        # Expected dendrogram
        dend_expected = np.array([(1, 4, 0.16853455, 2), (5, 0, 0.22227236, 3),
                                  (3, 6, 0.26530006, 4), (7, 2, 0.76268129, 5)])

        # Calculating simple correlation matrix for the TIC algorithm input
        etf_corr = etf_prices.corr()

        # Also testing on the tree that has a single element on the top level
        etf_classification_tree_alt = etf_classification_tree.copy()
        etf_classification_tree_alt['All'] = 0

        # Using the function
        dendrogram = tic._get_linkage_corr(etf_classification_tree, etf_corr)

        # Using the function on a tree with an extra level
        dendrogram_alt = tic._get_linkage_corr(etf_classification_tree_alt, etf_corr)

        # Testing that the obtained dendrogram is right
        # Transforming to DataFrames to get the same types
        np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dend_expected)),
                                       decimal=2)

        # Checking that the tree with an extra level returned the same result
        np.testing.assert_almost_equal(np.array(pd.DataFrame(dendrogram)), np.array(pd.DataFrame(dendrogram_alt)),
                                       decimal=2)

    @staticmethod
    def test_link_clusters():
        """
        Testing the transformation of linkage object from local linkage to global linkage
        """

        tic = TIC()

        # Creating a sample input
        lnk0 = np.array([])
        lnk1 = np.array([[1, 3, 0.10526126, 2],
                         [4, 2, 0.23105119, 3],
                         [0, 5, 0.40104189, 4]])
        items0 = [1010, 1020, 2030, 1040, 1030]
        items1 = [1010, 1020, 1040, 1030]

        # Expected link array
        link_expected = np.array([[1, 4, 0.10526126, 2],
                                  [5, 3, 0.23105119, 3],
                                  [0, 6, 0.40104189, 4]])

        # Calculating new link array
        link_new = tic._link_clusters(lnk0, lnk1, items0, items1)

        # Testing that the obtained new link is right
        np.testing.assert_almost_equal(link_new, link_expected, decimal=2)

    @staticmethod
    def test_update_dist():
        """
        Testing the update of the general distance matrix to take the new clusters into account
        """

        tic = TIC()

        # Creating a sample input
        dist0 = pd.DataFrame([[0.0, 0.595671], [0.595671, 0.0]], columns=[20, 10], index=[20, 10])
        lnk0 = np.array([[1, 4, 0.10526126, 2],
                         [5, 3, 0.23105119, 3],
                         [0, 6, 0.40104189, 4],
                         [7, 2, 0.59567056, 5]])
        lnk_ = np.array([[7, 2, 0.59567056, 5]])
        items0 = [1010, 1020, 20, 1040, 1030, 5, 6, 10, 8]

        # Alternative criterion - simple average
        alt_criterion = pd.DataFrame.mean

        # Expected distance matrix
        dist_expected = pd.DataFrame([0], columns=[8], index=[8])

        # Calculating distance matrix
        dist_new = tic._update_dist(dist0, lnk0, lnk_, items0, criterion=None)

        # Calculating the distance matrix with an alternative criterion
        dist_new_alt = tic._update_dist(dist0, lnk0, lnk_, items0, criterion=alt_criterion)

        # Testing that the obtained distance matrix is right
        np.testing.assert_almost_equal(np.array(dist_new), np.array(dist_expected), decimal=2)

        # Testing that the obtained distance matrix is right
        np.testing.assert_almost_equal(np.array(dist_new_alt), np.array(dist_expected), decimal=2)

    @staticmethod
    def test_get_atoms():
        """
        Testing the obtaining of the atoms included in an element from a linkage object
        """

        tic = TIC()

        # Creating a sample input
        lnk = np.array([(1, 4, 0.10526126, 2), (5, 3, 0.23105119, 3),
                        (0, 6, 0.40104189, 4), (7, 2, 0.59567056, 5)],
                       dtype=[('i0', int), ('i1', int), ('dist', float), ('num', int)])
        item = 5

        # Expected list of atoms
        atoms_expected = [1, 4]

        # Getting the atoms list
        atoms = tic._get_atoms(lnk, item)

        # Testing that the obtained atoms are right
        np.testing.assert_almost_equal(atoms, atoms_expected, decimal=2)

    def test_link2corr(self):
        """
        Test the process of deriving a correlation matrix from the linkage object
        """

        tic = TIC()

        # Creating a sample input
        etf_prices = self.returns_data.iloc[:, :5]
        etf_corr = etf_prices.corr()

        lnk = np.array([(1, 4, 0.10526126, 2), (5, 3, 0.23105119, 3),
                        (0, 6, 0.40104189, 4), (7, 2, 0.59567056, 5)],
                       dtype=[('i0', int), ('i1', int), ('dist', float), ('num', int)])
        lbls = etf_corr.index

        # Expected correlation matrix
        corr_expected = pd.DataFrame([[1, 0.678331, 0.290353, 0.678331, 0.678331],
                                      [0.678331, 1, 0.290353, 0.893231, 0.977840],
                                      [0.290353, 0.290353, 1, 0.290353, 0.290353],
                                      [0.678331, 0.893231, 0.290353, 1, 0.893231],
                                      [0.678331, 0.977840, 0.290353, 0.893231, 1]],
                                     index=lbls, columns=lbls)

        # Getting the correlation matrix
        corr = tic._link2corr(lnk, lbls)

        # Testing that the correlation matrix is right
        np.testing.assert_almost_equal(np.array(corr), np.array(corr_expected), decimal=2)

    def test_tic_correlation(self):
        """
        Test the calculation the Theory-Implies Correlation (TIC) matrix.
        """

        tic = TIC()

        # Taking the first 5 ETFs for test purposes
        etf_prices = self.returns_data.iloc[:, :5]
        etf_classification_tree = self.classification_tree.iloc[:5]

        # Calculating simple correlation matrix for the TIC algorithm input
        etf_corr = etf_prices.corr()

        # Calculating the relation of the number of observations to the number of elements
        tn_relation = etf_prices.shape[0] / etf_prices.shape[1]

        # Expected correlation matrix
        corr_expected = pd.DataFrame([[1, 0.72177129, -0.30629381, 0.7144813, 0.72177129],
                                      [0.72177129, 1, -0.30729469, 0.716816, 0.72412981],
                                      [-0.30629381, -0.30729469, 1, -0.30419096, -0.30729469],
                                      [0.7144813, 0.716816, -0.30419096, 1, 0.716816],
                                      [0.72177129, 0.72412981, -0.30729469, 0.716816, 1]],
                                     index=etf_corr.index, columns=etf_corr.index)

        # Getting the correlation matrix
        corr = tic.tic_correlation(etf_classification_tree, etf_corr, tn_relation=tn_relation, kde_bwidth=0.25)

        # Testing that the correlation matrix is right
        np.testing.assert_almost_equal(np.array(corr), np.array(corr_expected), decimal=2)

    def test_corr_dist(self):
        """
        Test the calculation of the correlation matrix distance
        """

        tic = TIC()

        # Taking the original correlation matrix and the TIC matrix
        etf_prices = self.returns_data.iloc[:, :5]
        etf_corr = etf_prices.corr()

        tic_corr = pd.DataFrame([[1, 0.72177129, -0.30629381, 0.7144813, 0.72177129],
                                 [0.72177129, 1, -0.30729469, 0.716816, 0.72412981],
                                 [-0.30629381, -0.30729469, 1, -0.30419096, -0.30729469],
                                 [0.7144813, 0.716816, -0.30419096, 1, 0.716816],
                                 [0.72177129, 0.72412981, -0.30729469, 0.716816, 1]],
                                index=etf_corr.index, columns=etf_corr.index)

        # Expected distance between the matrices
        dist_ecpected = 0.0130404424083

        # Calculating the distance between the correlation matrices
        distance = tic.corr_dist(etf_corr, tic_corr)

        # Testing that the calculated distance is right
        np.testing.assert_almost_equal(distance, dist_ecpected, decimal=2)
