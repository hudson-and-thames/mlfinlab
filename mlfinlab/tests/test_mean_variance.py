"""
Tests different algorithms in the Mean Variance class.
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


class TestMVO(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests the different functions of the Mean Variance Optimisation class
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_inverse_variance_solution(self):
        """
        Test the calculation of inverse-variance portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='inverse_variance', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_min_volatility_solution(self):
        """
        Test the calculation of inverse-variance portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_sharpe_solution(self):
        """
        Test the calculation of inverse-variance portfolio weights.
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
        Test the calculation of inverse-variance portfolio weights.
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
        Test the calculation of inverse-variance portfolio weights.
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
        Test MVO when we pass expected returns and covariance matrix as input.
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
        Test the calculation of weights when specific bounds are supplied.
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
        Test the calculation of weights when specific bounds are supplied.
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
        Test the calculation of weights when specific bounds are supplied.
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
        Test the calculation of inverse-variance portfolio weights.
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
        Test ValueError on passing unknown returns calculation string.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation(calculate_expected_returns='unknown_returns')
            mvo.allocate(asset_prices=self.data, asset_names=self.data.columns)

    def test_value_error_for_unknown_solution(self):
        # pylint: disable=invalid-name
        """
        Test ValueError on passing unknown solution string.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data, solution='ivp', asset_names=self.data.columns)

    def test_value_error_for_non_dataframe_input(self):
        # pylint: disable=invalid-name
        """
        Test ValueError on passing non-dataframe input.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data.values, solution='inverse_variance', asset_names=self.data.columns)

    def test_value_error_for_no_min_volatility_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for minimum volatility solution.
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
        Test ValueError when no optimal weights are found for maximum Sharpe solution.
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
        Test ValueError when no optimal weights are found for efficient risk solution.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='efficient_risk',
                         weight_bounds=(0.9, 1),
                         asset_names=self.data.columns)

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            data = self.data.reset_index()
            mvo.allocate(asset_prices=data, solution='inverse_variance', asset_names=self.data.columns)

    def test_resampling_asset_prices(self):
        """
        Test resampling of asset prices.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='inverse_variance', resample_by='B', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_all_inputs_none(self):
        """
        Test allocation when all inputs are None.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_names=self.data.columns)

    def test_no_asset_names(self):
        """
        Test MVO when not supplying a list of asset names.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_valuerror_with_no_asset_names(self):
        """
        Test ValueError when not supplying a list of asset names and no other input
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            expected_returns = ReturnsEstimation().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                     resample_by='W')
            covariance = ReturnsEstimation().calculate_returns(asset_prices=self.data, resample_by='W').cov()
            mvo.allocate(expected_asset_returns=expected_returns, covariance_matrix=covariance)
