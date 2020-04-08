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
        Test the calculation of minimum volatility portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='min_volatility', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

        # Check that the volatility is the minimum among all other portfolios
        for solution_string in {"inverse_variance", "max_return", "max_sharpe", "max_return_min_volatility",
                                "max_diversification", "max_decorrelation"}:
            mvo_ = MeanVarianceOptimisation()
            mvo_.allocate(asset_prices=self.data, solution=solution_string, asset_names=self.data.columns)
            assert mvo.portfolio_risk <= mvo_.portfolio_risk

    def test_max_return_solution(self):
        """
        Test the calculation of maximum expected return portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='max_return', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

        # Check that the return is the maximum among all other portfolios
        for solution_string in {"inverse_variance", "min_volatility", "max_sharpe", "max_return_min_volatility",
                                "max_diversification", "max_decorrelation"}:
            mvo_ = MeanVarianceOptimisation()
            mvo_.allocate(asset_prices=self.data, solution=solution_string, asset_names=self.data.columns)
            assert (mvo.portfolio_return > mvo_.portfolio_return or np.isclose(mvo.portfolio_return, mvo_.portfolio_return))

    def test_max_return_min_volatility_solution(self):
        """
        Test the calculation of maximum expected return and minimum volatility portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='max_return_min_volatility', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_sharpe_solution(self):
        """
        Test the calculation of maximum sharpe portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='max_sharpe', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

        # Check that the sharpe ratio is maximum
        for solution_string in {"inverse_variance", "min_volatility", "max_return", "max_return_min_volatility",
                         "max_diversification", "max_decorrelation"}:
            mvo_ = MeanVarianceOptimisation()
            mvo_.allocate(asset_prices=self.data, solution=solution_string, asset_names=self.data.columns)
            print(solution_string, mvo.portfolio_sharpe_ratio, mvo_.portfolio_sharpe_ratio)
            # assert mvo.portfolio_sharpe_ratio >= mvo_.portfolio_sharpe_ratio

    def test_min_volatility_for_target_return(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of minimum volatility-target return portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='efficient_risk', asset_names=self.data.columns, resample_by='W')
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_return_for_target_risk(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of maximum return-target risk portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='efficient_return', asset_names=self.data.columns, resample_by='W')
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_diversification(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of maximum diversification portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='max_diversification', asset_names=self.data.columns, resample_by='M')
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_decorrelation(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of maximum decorrelation portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data, solution='max_decorrelation', asset_names=self.data.columns, resample_by='W')
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_plotting_efficient_frontier(self):
        # pylint: disable=invalid-name, bad-continuation, protected-access
        """
        Test the plotting of the efficient frontier.
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
                     weight_bounds=['weights[0] <= 0.2'],
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
                     weight_bounds=['y[0] <= kappa * 0.5'],
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
                     weight_bounds=['weights[0] <= 0.3'],
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_efficient_return_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='efficient_return',
                     weight_bounds=['weights[0] <= 0.3'],
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_return_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='max_return',
                     weight_bounds=['weights[1] <= 1'],
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_decorrelation_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='max_decorrelation',
                     weight_bounds=['weights[0] <= 0.3'],
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_diversification_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='max_diversification',
                     weight_bounds=['weights[0] <= 0.3'],
                     asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_max_return_min_volatility_with_specific_weight_bounds(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of weights when specific bounds are supplied.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     solution='max_return_min_volatility',
                     weight_bounds=['weights[0] <= 0.3'],
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
