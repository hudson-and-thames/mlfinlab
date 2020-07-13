"""
Tests different algorithms in the Mean Variance class.
"""

import unittest
from unittest.mock import patch
from io import StringIO
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimators


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
        for solution_string in {"inverse_variance", "max_sharpe", "max_return_min_volatility",
                                "max_diversification", "max_decorrelation"}:
            mvo_ = MeanVarianceOptimisation()
            mvo_.allocate(asset_prices=self.data, solution=solution_string, asset_names=self.data.columns)
            assert mvo.portfolio_risk <= mvo_.portfolio_risk

    def test_max_return_min_volatility_solution(self):
        """
        Test the calculation of maximum expected return and minimum volatility portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data,
                     risk_aversion=50,
                     solution='max_return_min_volatility',
                     asset_names=self.data.columns)
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

    def test_min_volatility_for_target_return(self):
        # pylint: disable=invalid-name
        """
        Test the calculation of minimum volatility-target return portfolio weights.
        """

        mvo = MeanVarianceOptimisation()
        prices = self.data.resample('W').last()
        mvo.allocate(asset_prices=prices, solution='efficient_risk', asset_names=self.data.columns)
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
        mvo.allocate(asset_prices=self.data, solution='efficient_return', asset_names=self.data.columns)
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
        mvo.allocate(asset_prices=self.data, solution='max_diversification', asset_names=self.data.columns)
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
        mvo.allocate(asset_prices=self.data, solution='max_decorrelation', asset_names=self.data.columns)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_plotting_efficient_frontier(self):
        # pylint: disable=invalid-name, protected-access
        """
        Test the plotting of the efficient frontier.
        """

        mvo = MeanVarianceOptimisation()
        expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                 resample_by='W')
        covariance = ReturnsEstimators().calculate_returns(asset_prices=self.data, resample_by='W').cov()
        plot = mvo.plot_efficient_frontier(covariance=covariance,
                                           max_return=1.0,
                                           expected_asset_returns=expected_returns)
        assert plot.axes.xaxis.label._text == 'Volatility'
        assert plot.axes.yaxis.label._text == 'Return'
        assert len(plot._A) == 41

    def test_exception_in_plotting_efficient_frontier(self):
        # pylint: disable=invalid-name, protected-access
        """
        Test raising of exception when plotting the efficient frontier.
        """

        mvo = MeanVarianceOptimisation()
        expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                 resample_by='W')
        covariance = ReturnsEstimators().calculate_returns(asset_prices=self.data, resample_by='W').cov()
        plot = mvo.plot_efficient_frontier(covariance=covariance,
                                           max_return=1.0,
                                           expected_asset_returns=expected_returns)
        assert len(plot._A) == 41

    def test_mvo_with_input_as_returns_and_covariance(self):
        # pylint: disable=invalid-name
        """
        Test MVO when we pass expected returns and covariance matrix as input.
        """

        mvo = MeanVarianceOptimisation()
        expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data, resample_by='W')
        covariance = ReturnsEstimators().calculate_returns(asset_prices=self.data, resample_by='W').cov()
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
        mvo.allocate(asset_prices=self.data, solution='inverse_variance', asset_names=self.data.columns)
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

    def test_value_error_for_no_quadratic_utlity_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for max return-minimum volatility solution.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='max_return_min_volatility',
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

    def test_value_error_for_no_efficient_return_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for efficient return solution.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='efficient_return',
                         weight_bounds=(0.9, 1),
                         asset_names=self.data.columns)

    def test_value_error_for_no_max_diversification_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for max diversification solution.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='max_diversification',
                         weight_bounds=(0.9, 1),
                         asset_names=self.data.columns)

    def test_value_error_for_no_max_decorrelation_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for max decorrelation solution.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            mvo.allocate(asset_prices=self.data,
                         solution='max_decorrelation',
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
        mvo.allocate(asset_prices=self.data, solution='inverse_variance', asset_names=self.data.columns)
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

    def test_no_asset_names_by_passing_cov(self):
        """
        Test MVO when not supplying a list of asset names but passing covariance matrix as input
        """

        mvo = MeanVarianceOptimisation()
        expected_returns = ReturnsEstimators().calculate_exponential_historical_returns(asset_prices=self.data,
                                                                                        resample_by='W')
        covariance = ReturnsEstimators().calculate_returns(asset_prices=self.data, resample_by='W').cov()
        mvo.allocate(expected_asset_returns=expected_returns, covariance_matrix=covariance)
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
            expected_returns = ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                     resample_by='W')
            covariance = ReturnsEstimators().calculate_returns(asset_prices=self.data, resample_by='W').cov()
            mvo.allocate(expected_asset_returns=expected_returns, covariance_matrix=covariance.values)

    def test_portfolio_metrics(self):
        """
        Test the printing of portfolio metrics to stdout.
        """

        mvo = MeanVarianceOptimisation()
        mvo.allocate(asset_prices=self.data)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            mvo.get_portfolio_metrics()
            output = fake_out.getvalue().strip()
            self.assertTrue('Portfolio Return = 0.017362404155484328' in output)
            self.assertTrue('Portfolio Risk = 9.385801639141577e-06' in output)
            self.assertTrue('Portfolio Sharpe Ratio = -4.125045816381286' in output)

    def test_custom_objective_function(self):
        """
        Test custom portfolio objective and allocation constraints.
        """

        mvo = MeanVarianceOptimisation()
        custom_obj = 'cp.Minimize(risk)'
        constraints = ['cp.sum(weights) == 1', 'weights >= 0', 'weights <= 1']
        non_cvxpy_variables = {
            'num_assets': self.data.shape[1],
            'covariance': self.data.cov(),
            'expected_returns': ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                      resample_by='W')
        }
        cvxpy_variables = [
            'risk = cp.quad_form(weights, covariance)',
            'portfolio_return = cp.matmul(weights, expected_returns)'
        ]
        mvo.allocate_custom_objective(non_cvxpy_variables=non_cvxpy_variables,
                                      cvxpy_variables=cvxpy_variables,
                                      objective_function=custom_obj,
                                      constraints=constraints)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        assert mvo.asset_names == list(range(mvo.num_assets))
        assert mvo.portfolio_return == 0.012854555899642236
        assert  mvo.portfolio_risk == 3.0340907720046832
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_custom_objective_with_asset_names(self):
        """
        Test custom portfolio objective and constraints while providing a list of asset names.
        """

        mvo = MeanVarianceOptimisation()
        custom_obj = 'cp.Minimize(kappa)'
        constraints = ['cp.sum(weights) == 1', 'weights >= 0', 'weights <= 1']
        non_cvxpy_variables = {
            'num_assets': self.data.shape[1],
            'covariance': self.data.cov(),
            'asset_names': self.data.columns
        }
        cvxpy_variables = [
            'kappa = cp.quad_form(weights, covariance)',
        ]
        mvo.allocate_custom_objective(
            non_cvxpy_variables=non_cvxpy_variables,
            cvxpy_variables=cvxpy_variables,
            objective_function=custom_obj,
            constraints=constraints)
        weights = mvo.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        assert list(mvo.asset_names) == list(self.data.columns)
        assert mvo.portfolio_return is None
        assert mvo.portfolio_risk is None
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_value_error_for_custom_obj_optimal_weights(self):
        # pylint: disable=invalid-name
        """
        Test ValueError when no optimal weights are found for custom objective solution.
        """

        with self.assertRaises(ValueError):
            mvo = MeanVarianceOptimisation()
            custom_obj = 'cp.Minimize(risk - kappa)'
            constraints = ['cp.sum(weights) == 1', 'weights >= 0', 'weights <= 1']
            non_cvxpy_variables = {
                'num_assets': self.data.shape[1],
                'covariance': self.data.cov(),
                'expected_returns': ReturnsEstimators().calculate_mean_historical_returns(asset_prices=self.data,
                                                                                          resample_by='W')
            }
            cvxpy_variables = [
                'risk = cp.quad_form(weights, covariance)',
                'portfolio_return = cp.matmul(weights, expected_returns)',
                'kappa = cp.Variable(1)'
            ]
            mvo.allocate_custom_objective(non_cvxpy_variables=non_cvxpy_variables,
                                          cvxpy_variables=cvxpy_variables,
                                          objective_function=custom_obj,
                                          constraints=constraints)
