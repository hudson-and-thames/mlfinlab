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
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


class TestCLA(unittest.TestCase):
    # pylint: disable=too-many-public-methods
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
        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, asset_names=self.data.columns)
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

        cla = CLA(weight_bounds=([0]*self.data.shape[1], [1]*self.data.shape[1]), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, asset_names=self.data.columns)
        weights = cla.weights.values
        weights[weights <= 1e-15] = 0  # Convert very very small numbers to 0
        for turning_point in weights:
            assert (turning_point >= 0).all()
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)

    def test_cla_with_exponential_returns(self):
        """
        Test the calculation of CLA turning points using exponential returns
        """

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="exponential")
        cla.allocate(asset_prices=self.data, asset_names=self.data.columns)
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

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='max_sharpe', asset_names=self.data.columns)
        weights = cla.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cla_min_volatility(self):
        """
        Test the calculation for minimum volatility weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
        weights = cla.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_cla_efficient_frontier(self):
        """
        Test the calculation of the efficient frontier solution
        """

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='efficient_frontier', asset_names=self.data.columns)
        assert len(cla.efficient_frontier_means) == len(cla.efficient_frontier_sigma) and \
               len(cla.efficient_frontier_sigma) == len(cla.weights.values)
        assert cla.efficient_frontier_sigma[-1] <= cla.efficient_frontier_sigma[0] and \
               cla.efficient_frontier_means[-1] <= cla.efficient_frontier_means[0]  # higher risk = higher return

    def test_lambda_for_no_bounded_weights(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test the computation of lambda when there are no bounded weights
        """

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
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

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
        x, y = cla._free_bound_weight(free_weights=[1]*(cla.expected_returns.shape[0]+1))
        assert not x
        assert not y

    def test_expected_returns_equals_means(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test for condition when expected returns equal the mean value
        """

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
        data = self.data.copy()
        data.iloc[:, :] = 0.02320653
        cla._initialise(asset_prices=data, resample_by='B', expected_asset_returns=None, covariance_matrix=None)
        assert cla.expected_returns[-1, 0] == 1e-5

    def test_lambda_for_zero_matrices(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test the computation of lambda when there are no bounded weights. The method
        should return None, None
        """

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
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

        cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
        cla.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
        data = self.data.cov()
        data = data.values
        x, y = cla._compute_w(covar_f_inv=data,
                              covar_fb=data,
                              mean_f=cla.expected_returns,
                              w_b=None)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, float)

    def test_purge_excess(self):
        # pylint: disable=protected-access,invalid-name
        """
        Test purge number excess for very very small tolerance
        """

        with self.assertRaises(IndexError):
            cla = CLA(weight_bounds=(0, 1), calculate_expected_returns="mean")
            cla.allocate(asset_prices=self.data, solution='cla_turning_points', asset_names=self.data.columns)
            cla.weights = list(cla.weights.values)
            cla.weights = cla.weights*100
            cla._purge_num_err(tol=1e-18)

    def test_flag_true_for_purge_num_err(self):
        # pylint: disable=protected-access, no-self-use
        """
        Test whether the flag becomes True in the purge num error function
        """

        cla = CLA()
        cla.weights = [[1]]
        cla.lower_bounds = [100]
        cla.upper_bounds = [1]
        cla.lambdas = [[1]]
        cla.gammas = [[1]]
        cla.free_weights = [[1]]
        cla._purge_num_err(tol=1)
        assert not cla.weights
        assert not cla.lambdas
        assert not cla.gammas


    def test_value_error_for_unknown_solution(self):
        """
        Test ValueError on passing unknown solution string
        """

        with self.assertRaises(ValueError):
            cla = CLA()
            cla.allocate(asset_prices=self.data, solution='unknown_string', asset_names=self.data.columns)

    def test_value_error_for_non_dataframe_input(self):
        """
        Test ValueError on passing non-dataframe input
        """

        with self.assertRaises(ValueError):
            cla = CLA()
            cla.allocate(asset_prices=self.data.values, solution='cla_turning_points', asset_names=self.data.columns)

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date
        """

        with self.assertRaises(ValueError):
            cla = CLA()
            data = self.data.reset_index()
            cla.allocate(asset_prices=data, solution='cla_turning_points', asset_names=self.data.columns)

    def test_value_error_for_unknown_returns(self):
        """
        Test ValueError on passing unknown returns string
        """

        with self.assertRaises(ValueError):
            cla = CLA(calculate_expected_returns="unknown_returns")
            cla.allocate(asset_prices=self.data, solution='cla_turning_points', asset_names=self.data.columns)

    def test_resampling_asset_prices(self):
        """
        Test resampling of asset prices
        """

        cla = CLA()
        cla.allocate(asset_prices=self.data, resample_by='B', solution='min_volatility', asset_names=self.data.columns)
        weights = cla.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_all_inputs_none(self):
        """
        Test allocation when all inputs are None
        """

        with self.assertRaises(ValueError):
            cla = CLA()
            cla.allocate(asset_names=self.data.columns)

    def test_cla_with_input_as_returns_and_covariance(self):
        # pylint: disable=invalid-name
        """
        Test CLA when we pass expected returns and covariance matrix as input
        """

        cla = CLA()
        expected_returns = ReturnsEstimation().calculate_mean_historical_returns(asset_prices=self.data)
        covariance = ReturnsEstimation().calculate_returns(asset_prices=self.data).cov()
        cla.allocate(covariance_matrix=covariance,
                     expected_asset_returns=expected_returns,
                     asset_names=self.data.columns)
        weights = cla.weights.values
        weights[weights <= 1e-15] = 0  # Convert very very small numbers to 0
        for turning_point in weights:
            assert (turning_point >= 0).all()
            assert len(turning_point) == self.data.shape[1]
            np.testing.assert_almost_equal(np.sum(turning_point), 1)


class TestHRP(unittest.TestCase):
    # pylint: disable=too-many-public-methods
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
        hrp.allocate(asset_prices=self.data, asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_hrp_with_shrinkage(self):
        """
        Test the weights calculated by HRP algorithm with covariance shrinkage
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, use_shrinkage=True, asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_dendrogram_plot(self):
        """
        Test if dendrogram plot object is correctly rendered
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, use_shrinkage=True, asset_names=self.data.columns)
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
        hrp.allocate(asset_prices=self.data, asset_names=self.data.columns)
        assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17,
                                       12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]

    def test_value_error_for_non_dataframe_input(self):
        """
        Test ValueError on passing non-dataframe input
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            hrp.allocate(asset_prices=self.data.values, asset_names=self.data.columns)

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            data = self.data.reset_index()
            hrp.allocate(asset_prices=data, asset_names=self.data.columns)

    def test_resampling_asset_prices(self):
        """
        Test resampling of asset prices
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, resample_by='B', asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_all_inputs_none(self):
        """
        Test allocation when all inputs are None
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            hrp.allocate(asset_names=self.data.columns)

    def test_hrp_with_input_as_returns(self):
        """
        Test HRP when passing asset returns dataframe as input
        """

        hrp = HierarchicalRiskParity()
        returns = ReturnsEstimation().calculate_returns(asset_prices=self.data)
        hrp.allocate(asset_returns=returns, asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_hrp_with_input_as_covariance_matrix(self):
        """
        Test HRP when passing a covariance matrix as input
        """

        hrp = HierarchicalRiskParity()
        returns = ReturnsEstimation().calculate_returns(asset_prices=self.data)
        hrp.allocate(asset_names=self.data.columns, covariance_matrix=returns.cov())
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

class TestMVO(unittest.TestCase):
    # pylint: disable=too-many-public-methods
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

    def test_inverse_variance_solution(self):
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='inverse_variance', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_min_volatility_solution(self):
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_sharpe_solution(self):
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='max_sharpe', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_min_volatility_with_target_return(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='efficient_risk', asset_names=self.data.columns, resample_by='W')
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_plotting_efficient_frontier(self):
        # pylint: disable=invalid-name, bad-continuation, protected-access
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation()
        expected_returns = ReturnsEstimation().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                 resample_by='W')
        covariance = ReturnsEstimation().calculate_returns(asset_prices=self.data, resample_by='W').cov()
        plot = mvo.plot_efficient_frontier(covariance=covariance,
                                           num_assets=self.data.shape[1],
                                           expected_asset_returns=expected_returns)
        assert plot.axes.xaxis.label._text == 'Volatility'
        assert plot.axes.yaxis.label._text == 'Return'
        assert len(plot._A) == 100

    def test_mvo_with_input_as_returns_and_covariance(self):
        # pylint: disable=invalid-name, bad-continuation
        """
        Test MVO when we pass expected returns and covariance matrix as input
        """

        mvo = MeanVarianceOptimisation()
        expected_returns = ReturnsEstimation().calculate_mean_historical_returns(asset_prices=self.data, resample_by='W')
        covariance = ReturnsEstimation().calculate_returns(asset_prices=self.data, resample_by='W').cov()
        mvo.allocate(covariance_matrix=covariance,
                     expected_asset_returns=expected_returns,
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_min_volatility_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='min_volatility',
                     weight_bounds={0:(0.3, 1)},
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_sharpe_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='max_sharpe',
                     weight_bounds={0: (0.3, 1)},
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_efficient_risk_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='efficient_risk',
                     target_return=0.01,
                     weight_bounds={0: (0.3, 1)},
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_mvo_with_exponential_returns(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of inverse-variance portfolio weights
        """

        mvo = MeanVarianceOptimisation(calculate_expected_returns='exponential')
        mvo.allocate(asset_prices=self.data, resample_by='B', solution='inverse_variance', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_unknown_returns_calculation(self):
        # pylint: disable=invalid-name
        """
        Test ValueError on passing unknown returns calculation string
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation(calculate_expected_returns='unknown_returns')
            mvo.allocate(asset_prices=self.data, asset_names=self.data.columns)

    def test_value_error_for_unknown_solution(self):
        # pylint: disable=invalid-name
        """
        Test ValueError on passing unknown solution string
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data, solution='ivp', asset_names=self.data.columns)

    def test_value_error_for_non_dataframe_input(self):
        # pylint: disable=invalid-name
        """
        Test ValueError on passing non-dataframe input
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data.values, solution='inverse_variance', asset_names=self.data.columns)

    def test_value_error_for_no_min_volatility_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for minimum volatility solution
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='min_volatility',
                         weight_bounds=(0.9, 1),
                         asset_names=self.data.columns)

    def test_value_error_for_no_max_sharpe_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for maximum Sharpe solution
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='max_sharpe',
                         weight_bounds=(0.9, 1),
                         asset_names=self.data.columns)

    def test_value_error_for_no_efficient_risk_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for efficient risk solution
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='efficient_risk',
                         weight_bounds=(0.9, 1),
                         asset_names=self.data.columns)

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            data = self.data.reset_index()
            mvo.allocate(asset_prices=data, solution='inverse_variance', asset_names=self.data.columns)

    def test_resampling_asset_prices(self):
        """
        Test resampling of asset prices
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='inverse_variance', resample_by='B', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_all_inputs_none(self):
        """
        Test allocation when all inputs are None
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_names=self.data.columns)
