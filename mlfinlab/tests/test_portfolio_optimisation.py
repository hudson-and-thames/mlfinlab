"""
Tests the different portfolio optimisation algorithms
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.cla import CLA
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation


class TestCLA(unittest.TestCase):
    """
    Tests different functions of the CLA class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_cla_with_mean_returns(self):
        """
        Test the calculation of CLA turning points using mean returns
        """

        self.data.iloc[1:10, :] = 40
        self.data.iloc[11:20, :] = 50
        self.data.iloc[21, :] = 100
        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data)
        weights = cla.weights.values
        weights[weights <= 1e-15] = 0 # Convert very very small numbers to 0
        for turning_point in weights:
            assert (turning_point >= 0).all()
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)

    def test_cla_with_weight_bounds_as_lists(self):
        """
        Test the calculation of CLA turning points when we pass the weight bounds as a list
        instead of just lower and upper bound value
        """

        cla = CLA(weight_bounds=([0]*self.data.shape[1], [1]*self.data.shape[1]), calculate_returns="mean")
        cla.allocate(asset_prices=self.data)
        weights = cla.weights.values
        weights[weights <= 1e-15] = 0 # Convert very very small numbers to 0
        for turning_point in weights:
            assert (turning_point >= 0).all()
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)

    def test_cla_with_exponential_returns(self):
        """
        Test the calculation of CLA turning points using exponential returns
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="exponential")
        cla.allocate(asset_prices=self.data)
        weights = cla.weights.values
        weights[weights <= 1e-15] = 0 # Convert very very small numbers to 0
        for turning_point in weights:
            assert (turning_point >= 0).all()
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)

    def test_cla_max_sharpe(self):
        """
        Test the calculation of maximum sharpe ratio weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='max_sharpe')
        weights = cla.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cla_min_volatility(self):
        """
        Test the calculation for minimum volatility weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        weights = cla.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cla_efficient_frontier(self):
        """
        Test the calculation of the efficient frontier solution
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='efficient_frontier')
        assert len(cla.efficient_frontier_means) == len(cla.efficient_frontier_sigma) and \
               len(cla.efficient_frontier_sigma) == len(cla.weights.values)
        assert cla.efficient_frontier_sigma[-1] <= cla.efficient_frontier_sigma[0] and \
               cla.efficient_frontier_means[-1] <= cla.efficient_frontier_means[0]  # higher risk = higher return

    def test_lambda_for_no_bounded_weights(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test the computation of lambda when there are no bounded weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        data = self.data.cov()
        data = data.values
        x, y = cla._compute_lambda(covar_f_inv=data,
                                   covar_fb=data,
                                   mean_f=cla.expected_returns,
                                   w_b=None,
                                   asset_index=1,
                                   b_i=[[0], [1]])
        assert isinstance(x, float)
        assert isinstance(y, int)

    def test_free_bound_weights(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test the method of freeing bounded weights when free-weights is None
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        x, y = cla._free_bound_weight(free_weights=[1]*(cla.expected_returns.shape[0]+1))
        assert not x
        assert not y

    def test_expected_returns_equals_means(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test for condition when expected returns equal the mean value
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        data = self.data.copy()
        data.iloc[:, :] = 0.02320653
        cla._initialise(asset_prices=data, resample_by='B')
        assert cla.expected_returns[-1, 0] == 1e-5

    def test_lambda_for_zero_matrices(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test the computation of lambda when there are no bounded weights. The method
        should return None, None
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        data = self.data.cov()
        data = data.values
        data[:, :] = 0
        x, y = cla._compute_lambda(covar_f_inv=data,
                                   covar_fb=data,
                                   mean_f=cla.expected_returns,
                                   w_b=None,
                                   asset_index=1,
                                   b_i=[[0], [1]])
        assert not x
        assert not y

    def test_w_for_no_bounded_weights(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test the computation of weights (w) when there are no bounded weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility')
        data = self.data.cov()
        data = data.values
        x, y = cla._compute_w(covar_f_inv=data,
                              covar_fb=data,
                              mean_f=cla.expected_returns,
                              w_b=None)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

    def test_purge_num_excess(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test purge number excess for very very small tolerance
        """

        with self.assertRaises(IndexError):
            cla = CLA(weight_bounds=(0, 1), calculate_returns="mean")
            cla.allocate(asset_prices=self.data, solution='cla_turning_points')
            cla.weights = list(cla.weights.values)
            cla.weights = cla.weights*100
            cla._purge_num_err(tol=1e-18)

    def test_value_error_for_unknown_solution(self):
        """
        Test ValueError on passing unknown solution string
        """

        with self.assertRaises(ValueError):
            cla = CLA()
            cla.allocate(asset_prices=self.data, solution='unknown_string')

    def test_value_error_for_non_dataframe_input(self):
        """
        Test ValueError on passing non-dataframe input
        """

        with self.assertRaises(ValueError):
            cla = CLA()
            cla.allocate(asset_prices=self.data.values, solution='cla_turning_points')

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date
        """

        with self.assertRaises(ValueError):
            cla = CLA()
            data = self.data.reset_index()
            cla.allocate(asset_prices=data, solution='cla_turning_points')

    def test_value_error_for_unknown_returns(self):
        """
        Test ValueError on passing unknown returns string
        """

        with self.assertRaises(ValueError):
            cla = CLA(calculate_returns="unknown_returns")
            cla.allocate(asset_prices=self.data, solution='cla_turning_points')

class TestHRP(unittest.TestCase):
    """
    Tests different functions of the HRP algorithm class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_hrp(self):
        """
        Test the weights calculated by the HRP algorithm - if all the weights are positive and
        their sum is equal to 1.
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_hrp_with_shrinkage(self):
        """
        Test the weights calculated by HRP algorithm with covariance shrinkage
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, use_shrinkage=True)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_dendrogram_plot(self):
        """
        Test if dendrogram plot object is correctly rendered
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, use_shrinkage=True)
        dendrogram = hrp.plot_clusters(assets=self.data.columns)
        assert dendrogram.get('icoord')
        assert dendrogram.get('dcoord')
        assert dendrogram.get('ivl')
        assert dendrogram.get('leaves')
        assert dendrogram.get('color_list')

    def test_quasi_diagnalization(self):
        """
        Test the quasi-diagnalisation step of HRP algorithm
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17,
                                       12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]

    def test_value_error_for_non_dataframe_input(self):
        """
        Test ValueError on passing non-dataframe input
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            hrp.allocate(asset_prices=self.data.values)

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            data = self.data.reset_index()
            hrp.allocate(asset_prices=data)

class TestMVO(unittest.TestCase):
    """
    Tests the different functions of the Mean Variance Optimisation class
    """

    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_inverse_variance(self):
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='inverse_variance')
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_value_error_for_unknown_solution(self):
        """
        Test ValueError on passing unknown solution string
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data, solution='ivp')

    def test_value_error_for_non_dataframe_input(self):
        """
        Test ValueError on passing non-dataframe input
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data.values, solution='inverse_variance')

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            data = self.data.reset_index()
            mvo.allocate(asset_prices=data, solution='inverse_variance')
