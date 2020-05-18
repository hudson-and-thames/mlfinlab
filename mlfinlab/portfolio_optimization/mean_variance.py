# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


class MeanVarianceOptimisation:
    """
    This class implements some classic mean-variance optimisation techniques for calculating the efficient frontier solutions.
    With the help of quadratic optimisers, users can generate optimal portfolios for different objective functions. Currently
    solutions to the following portfolios can be generated:
        1. Inverse Variance
        2. Maximum Sharpe
        3. Minimum Volatility
        4. Efficient Risk
        5. Maximum Return - Minimum Volatility
        6. Efficient Return
        7. Maximum Diversification
        8. Maximum Decorrelation
        9. Custom Objective Function
    """

    def __init__(self, calculate_expected_returns='mean', risk_free_rate=0.03):
        """
        Constructor.

        :param calculate_expected_returns: (str) The method to use for calculation of expected returns.
                                                 Currently supports "mean" and "exponential".
        """

        self.weights = list()
        self.portfolio_risk = None
        self.portfolio_return = None
        self.portfolio_sharpe_ratio = None
        self.calculate_expected_returns = calculate_expected_returns
        self.returns_estimator = ReturnsEstimation()
        self.risk_estimators = RiskEstimators()
        self.weight_bounds = (0, 1)
        self.risk_free_rate = risk_free_rate

    def allocate(self,
                 asset_names=None,
                 asset_prices=None,
                 expected_asset_returns=None,
                 covariance_matrix=None,
                 solution='inverse_variance',
                 target_return=0.2,
                 target_risk=0.01,
                 risk_aversion=10,
                 weight_bounds=None,
                 resample_by=None):
        # pylint: disable=invalid-name, too-many-branches, bad-continuation, too-many-arguments
        """
        Calculate the portfolio asset allocations using the method specified.

        :param asset_names: (list) a list of strings containing the asset names.
        :param asset_prices: (pd.Dataframe) A dataframe of historical asset prices (daily close).
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns (sigma).
        :param solution: (str) The type of solution/algorithm to use to calculate the weights.
                               Currently supported solution strings - inverse_variance, min_volatility, max_sharpe,
                               efficient_risk, max_return_min_volatility, max_diversification, efficient_return
                               and max_decorrelation.
        :param target_return: (float) Target return of the portfolio.
        :param target_risk: (float) Target risk of the portfolio.
        :param risk_aversion: (float) Quantifies the risk averse nature of the investor - a higher value means
                                      more risk averse and vice-versa.
        :param weight_bounds: (dict/tuple) Can be either a single tuple of upper and lower bounds
                                           for all portfolio weights or a list of strings with each string representing
                                           an inequality on the weights. For e.g. to bound the weight of the 3rd asset
                                           pass the following weight bounds: ['weights[2] <= 0.3', 'weights[2] >= 0.1'].
        :param resample_by: (str) Specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling.
        """

        asset_names = self._error_checks(asset_names, asset_prices, expected_asset_returns, covariance_matrix)

        # Weight bounds
        if weight_bounds is not None:
            self.weight_bounds = weight_bounds

        # Calculate the expected asset returns and covariance matrix if not given by the user
        expected_asset_returns, cov = self._calculate_estimators(asset_names,
                                                                 asset_prices,
                                                                 expected_asset_returns,
                                                                 covariance_matrix,
                                                                 resample_by)

        if solution == 'inverse_variance':
            self.weights = self._inverse_variance(covariance=cov)
            self.portfolio_risk = np.dot(self.weights, np.dot(cov.values, self.weights.T))
            self.portfolio_return = np.dot(self.weights, expected_asset_returns)[0]
        elif solution == 'min_volatility':
            self._min_volatility(covariance=cov,
                                 expected_returns=expected_asset_returns,
                                 num_assets=len(asset_names))
        elif solution == 'max_return_min_volatility':
            self._max_return_min_volatility(covariance=cov,
                                            expected_returns=expected_asset_returns,
                                            risk_aversion=risk_aversion,
                                            num_assets=len(asset_names))
        elif solution == 'max_sharpe':
            self._max_sharpe(covariance=cov,
                             expected_returns=expected_asset_returns,
                             num_assets=len(asset_names))
        elif solution == 'efficient_risk':
            self._min_volatility_for_target_return(covariance=cov,
                                                   expected_returns=expected_asset_returns,
                                                   target_return=target_return,
                                                   num_assets=len(asset_names))
        elif solution == 'efficient_return':
            self._max_return_for_target_risk(covariance=cov,
                                             expected_returns=expected_asset_returns,
                                             target_risk=target_risk,
                                             num_assets=len(asset_names))
        elif solution == 'max_diversification':
            self._max_diversification(covariance=cov,
                                      expected_returns=expected_asset_returns,
                                      num_assets=len(asset_names))
        elif solution == 'max_decorrelation':
            self._max_decorrelation(covariance=cov,
                                    expected_returns=expected_asset_returns,
                                    num_assets=len(asset_names))
        else:
            raise ValueError("Unknown solution string specified. Supported solutions - "
                             "inverse_variance, min_volatility, max_sharpe, efficient_risk"
                             "max_return_min_volatility, max_diversification, efficient_return and max_decorrelation")

        # Calculate the portfolio sharpe ratio
        self.portfolio_sharpe_ratio = ((self.portfolio_return - self.risk_free_rate) / (self.portfolio_risk ** 0.5))

        # Do some post-processing of the weights
        self._post_process_weights()
        self.weights = pd.DataFrame(self.weights)
        self.weights.index = asset_names
        self.weights = self.weights.T

    def allocate_custom_objective(self,
                                  custom_objective,
                                  asset_names=None,
                                  asset_prices=None,
                                  expected_asset_returns=None,
                                  covariance_matrix=None,
                                  target_return=0.2,
                                  target_risk=0.01,
                                  risk_aversion=10,
                                  resample_by=None):
        # pylint: disable=bad-continuation, eval-used, too-many-locals
        """
        Create a portfolio using custom objective and constraints.

        :param custom_objective: (dict) A custom objective function with custom constraints. You need to write it in the form
                                        expected by cvxpy. The objective will be a single string while the constraints can be a
                                        list of strings specifying the constraints. For e.g. {'objective': 'cp.Maximisie(
                                        expected_asset_returns)', 'constraints': ['weights >= 0', 'weights <= 1']}.
        :param asset_names: (list) A list of strings containing the asset names.
        :param asset_prices: (pd.Dataframe) A dataframe of historical asset prices (daily close).
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns (sigma).
        :param target_return: (float) Target return of the portfolio.
        :param target_risk: (float) Target risk of the portfolio.
        :param risk_aversion: (float) Quantifies the risk averse nature of the investor - a higher value means
                                      more risk averse and vice-versa.
        :param resample_by: (str) Specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling.
        """

        asset_names = self._error_checks(asset_names, asset_prices, expected_asset_returns, covariance_matrix)

        # Calculate the expected asset returns and covariance matrix if not given by the user
        expected_asset_returns, cov = self._calculate_estimators(asset_names,
                                                                 asset_prices,
                                                                 expected_asset_returns,
                                                                 covariance_matrix,
                                                                 resample_by)

        num_assets = len(asset_names)
        weights = cp.Variable(num_assets)
        weights.value = np.array([1 / num_assets] * num_assets)
        risk = cp.quad_form(weights, cov)
        portfolio_return = cp.matmul(weights, expected_asset_returns)

        # Optimisation objective and constraints
        objective, constraints = custom_objective['objective'], custom_objective['constraints']
        allocation_objective = eval(objective)
        allocation_constraints = []
        for constraint in constraints:
            allocation_constraints.append(eval(constraint))

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve(warm_start=True)
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')

        self.weights = weights.value
        self.portfolio_risk = risk.value
        self.portfolio_return = portfolio_return.value[0]

        # Calculate the portfolio sharpe ratio
        self.portfolio_sharpe_ratio = ((self.portfolio_return - self.risk_free_rate) / (self.portfolio_risk ** 0.5))

        # Do some post-processing of the weights
        self._post_process_weights()
        self.weights = pd.DataFrame(self.weights)
        self.weights.index = asset_names
        self.weights = self.weights.T

    def get_portfolio_metrics(self):
        """
        Prints the portfolio metrics - return, risk and Sharpe Ratio.
        """

        print("Portfolio Return = %s" % self.portfolio_return)
        print("Portfolio Risk = %s" % self.portfolio_risk)
        print("Portfolio Sharpe Ratio = %s" % self.portfolio_risk)

    def plot_efficient_frontier(self,
                                covariance,
                                expected_asset_returns,
                                num_assets,
                                min_return=0,
                                max_return=0.4,
                                risk_free_rate=0.05):
        # pylint: disable=bad-continuation, broad-except
        """
        Plot the Markowitz efficient frontier.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param num_assets: (int) Number of assets in the portfolio.
        :param min_return: (float) Minimum target return.
        :param max_return: (float) Maximum target return.
        :param risk_free_rate: (float) The rate of return for a risk-free asset.
        """

        expected_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))
        volatilities = []
        returns = []
        sharpe_ratios = []
        for portfolio_return in np.linspace(min_return, max_return, 100):
            try:
                self._min_volatility_for_target_return(covariance=covariance,
                                                        expected_returns=expected_returns,
                                                        target_return=portfolio_return,
                                                        num_assets=num_assets)
                volatilities.append(self.portfolio_risk)
                returns.append(portfolio_return)
                sharpe_ratios.append((portfolio_return - risk_free_rate) / (self.portfolio_risk ** 0.5 + 1e-16))
            except Exception:
                continue
        max_sharpe_ratio_index = sharpe_ratios.index(max(sharpe_ratios))
        min_volatility_index = volatilities.index(min(volatilities))
        figure = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(volatilities[max_sharpe_ratio_index],
                    returns[max_sharpe_ratio_index],
                    marker='*',
                    color='g',
                    s=400,
                    label='Maximum Sharpe Ratio')
        plt.scatter(volatilities[min_volatility_index],
                    returns[min_volatility_index],
                    marker='*',
                    color='r',
                    s=400,
                    label='Minimum Volatility')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.legend(loc='upper left')
        return figure

    @staticmethod
    def _error_checks(asset_names, asset_prices, expected_asset_returns, covariance_matrix):
        """
        Some initial error checks on the inputs.

        :param asset_names: (list) A list of strings containing the asset names.
        :param asset_prices: (pd.Dataframe) A dataframe of historical asset prices (daily close).
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns (sigma).
        :return: (list) List of asset names in the portfolio.
        """

        if asset_prices is None and (expected_asset_returns is None or covariance_matrix is None):
            raise ValueError("You need to supply either raw prices or expected returns "
                             "and a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        if asset_names is None:
            if asset_prices is not None:
                asset_names = asset_prices.columns
            elif covariance_matrix is not None and isinstance(covariance_matrix, pd.DataFrame):
                asset_names = covariance_matrix.columns
            else:
                raise ValueError("Please provide a list of asset names")

        return asset_names

    def _calculate_estimators(self,
                              asset_names,
                              asset_prices,
                              expected_asset_returns,
                              covariance_matrix,
                              resample_by):
        # pylint: disable=bad-continuation
        """
        Calculate the expected returns and covariance matrix of assets in the portfolio.

        :param asset_names: (list) A list of strings containing the asset names.
        :param asset_prices: (pd.Dataframe) A dataframe of historical asset prices (daily close).
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.Dataframe/numpy matrix) User supplied covariance matrix of asset returns (sigma).
        :param resample_by: (str) Specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling.
        :return: (np.array, pd.DataFrame) Expected asset returns and covariance matrix.
        """

        # Calculate the expected returns if the user does not supply any returns
        if expected_asset_returns is None:
            if self.calculate_expected_returns == "mean":
                expected_asset_returns = self.returns_estimator.calculate_mean_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            elif self.calculate_expected_returns == "exponential":
                expected_asset_returns = self.returns_estimator.calculate_exponential_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")
        expected_asset_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices, resample_by=resample_by)
            covariance_matrix = returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

        return expected_asset_returns, cov

    def _post_process_weights(self):
        """
        Check weights for very small numbers and numbers close to 1. A final post-processing of weights produced by the
        optimisation procedures.
        """

        # Round weights which are very very small negative numbers (e.g. -4.7e-16) to 0
        self.weights[self.weights < 0] = 0

        # If any of the weights is very close to one, we convert it to 1 and set the other asset weights to 0.
        if True in set(np.isclose(self.weights, 1)):
            almost_one_index = np.isclose(self.weights, 1)
            self.weights[almost_one_index] = 1
            self.weights[np.logical_not(almost_one_index)] = 0

    @staticmethod
    def _inverse_variance(covariance):
        """
        Calculate weights using inverse-variance allocation.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :return: (np.array) Array of portfolio weights.
        """

        ivp = 1. / np.diag(covariance)
        ivp /= ivp.sum()
        return ivp

    def _min_volatility(self, covariance, expected_returns, num_assets):
        # pylint: disable=eval-used
        """
        Compute minimum volatility portfolio allocation.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (np.array, float, float) Portfolio weights, risk value and return value.
        """

        weights = cp.Variable(num_assets)
        weights.value = np.array([1 / num_assets] * num_assets)
        risk = cp.quad_form(weights, covariance)
        portfolio_return = cp.matmul(weights, expected_returns)

        # Optimisation objective and constraints
        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum(weights) == 1,
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    weights >= self.weight_bounds[0],
                    weights <= min(self.weight_bounds[1], 1)
                ]
            )
        else:
            for inequality in self.weight_bounds:
                allocation_constraints.append(eval(inequality))

            # Add the hard-boundaries for weights.
            allocation_constraints.extend(
                [
                    weights <= 1,
                    weights >= 0
                ]
            )

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve(warm_start=True)
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')

        self.weights = weights.value
        self.portfolio_risk = risk.value
        self.portfolio_return = portfolio_return.value[0]

    def _max_return_min_volatility(self, covariance, expected_returns, risk_aversion, num_assets):
        # pylint: disable=eval-used
        """
        Calculate maximum return-minimum volatility portfolio allocation.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param risk_aversion: (float) Quantifies the risk averse nature of the investor - a higher value means
                           more risk averse and vice-versa.
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (np.array, float, float) Portfolio weights, risk value and return value.
        """

        weights = cp.Variable(num_assets)
        weights.value = np.array([1 / num_assets] * num_assets)
        portfolio_return = cp.matmul(weights, expected_returns)
        risk = cp.quad_form(weights, covariance)

        # Optimisation objective and constraints
        allocation_objective = cp.Minimize(risk_aversion * risk - portfolio_return)
        allocation_constraints = [
            cp.sum(weights) == 1
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    weights >= self.weight_bounds[0],
                    weights <= min(self.weight_bounds[1], 1)
                ]
            )
        else:
            for inequality in self.weight_bounds:
                allocation_constraints.append(eval(inequality))

            # Add the hard-boundaries for weights.
            allocation_constraints.extend(
                [
                    weights <= 1,
                    weights >= 0
                ]
            )

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve(warm_start=True)
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')

        self.weights = weights.value
        self.portfolio_risk = risk.value
        self.portfolio_return = portfolio_return.value[0]

    def _max_sharpe(self, covariance, expected_returns, num_assets):
        # pylint: disable=invalid-name,eval-used
        """
        Compute maximum Sharpe portfolio allocation.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (np.array, float, float) Portfolio weights, risk value and return value.
        """

        y = cp.Variable(num_assets)
        y.value = np.array([1 / num_assets] * num_assets)
        kappa = cp.Variable(1)
        risk = cp.quad_form(y, covariance)
        weights = y / kappa
        portfolio_return = cp.matmul(weights, expected_returns)

        # Optimisation objective and constraints
        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum((expected_returns - self.risk_free_rate).T @ y) == 1,
            cp.sum(y) == kappa,
            kappa >= 0
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    y >= kappa * self.weight_bounds[0],
                    y <= kappa * self.weight_bounds[1]
                ]
            )
        else:
            for inequality in self.weight_bounds:
                allocation_constraints.append(eval(inequality))

            # Add the hard-boundaries for weights.
            allocation_constraints.extend(
                [
                    y <= kappa,
                    y >= 0
                ]
            )

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve(warm_start=True)
        if y.value is None or kappa.value is None:
            raise ValueError('No optimal set of weights found.')

        self.weights = weights.value
        self.portfolio_risk = risk.value
        self.portfolio_return = portfolio_return.value[0]

    def _min_volatility_for_target_return(self, covariance, expected_returns, target_return, num_assets):
        # pylint: disable=eval-used
        """
        Calculate minimum volatility portfolio for a given target return.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param target_return: (float) Target return of the portfolio.
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (np.array, float, float) Portfolio weights, risk value and return value.
        """

        weights = cp.Variable(num_assets)
        weights.value = np.array([1 / num_assets] * num_assets)
        risk = cp.quad_form(weights, covariance)
        portfolio_return = cp.matmul(weights, expected_returns)

        # Optimisation objective and constraints
        allocation_objective = cp.Minimize(risk)
        allocation_constraints = [
            cp.sum(weights) == 1,
            portfolio_return >= target_return,
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    weights >= self.weight_bounds[0],
                    weights <= min(self.weight_bounds[1], 1)
                ]
            )
        else:
            for inequality in self.weight_bounds:
                allocation_constraints.append(eval(inequality))

            # Add the hard-boundaries for weights.
            allocation_constraints.extend(
                [
                    weights <= 1,
                    weights >= 0
                ]
            )

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve()
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')

        self.weights = weights.value
        self.portfolio_risk = risk.value
        self.portfolio_return = target_return

    def _max_return_for_target_risk(self, covariance, expected_returns, target_risk, num_assets):
        # pylint: disable=eval-used
        """
        Calculate maximum return for a given target volatility/risk.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param target_risk: (float) Target risk of the portfolio.
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (np.array, float, float) Portfolio weights, risk value and return value.
        """

        weights = cp.Variable(num_assets)
        weights.value = np.array([1 / num_assets] * num_assets)
        portfolio_return = cp.matmul(weights, expected_returns)
        risk = cp.quad_form(weights, covariance)

        # Optimisation objective and constraints
        allocation_objective = cp.Maximize(portfolio_return)
        allocation_constraints = [
            cp.sum(weights) == 1,
            risk <= target_risk
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    weights >= self.weight_bounds[0],
                    weights <= min(self.weight_bounds[1], 1)
                ]
            )
        else:
            for inequality in self.weight_bounds:
                allocation_constraints.append(eval(inequality))

            # Add the hard-boundaries for weights.
            allocation_constraints.extend(
                [
                    weights <= 1,
                    weights >= 0
                ]
            )

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve()
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')

        self.weights = weights.value
        self.portfolio_risk = target_risk
        self.portfolio_return = portfolio_return.value[0]

    def _max_diversification(self, covariance, expected_returns, num_assets):
        """
        Calculate the maximum diversified portfolio.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (np.array, float, float) Portfolio weights, risk value and return value.
        """

        self._max_decorrelation(covariance, expected_returns, num_assets)

        # Divide weights by individual asset volatilities
        self.weights /= np.diag(covariance)

        # Standardize weights
        self.weights /= np.sum(self.weights)

        portfolio_return = np.dot(expected_returns.T, self.weights)[0]
        risk = np.dot(self.weights, np.dot(covariance, self.weights.T))

        self.portfolio_risk = risk
        self.portfolio_return = portfolio_return

    def _max_decorrelation(self, covariance, expected_returns, num_assets):
        # pylint: disable=eval-used
        """
        Calculate the maximum decorrelated portfolio.

        :param covariance: (pd.Dataframe) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param num_assets: (int) Number of assets in the portfolio.
        :return: (np.array, float, float) Portfolio weights, risk value and return value.
        """

        weights = cp.Variable(num_assets)
        weights.value = np.array([1 / num_assets] * num_assets)
        risk = cp.quad_form(weights, covariance)
        portfolio_return = cp.matmul(weights, expected_returns)
        corr = self.risk_estimators.cov_to_corr(covariance)
        portfolio_correlation = cp.quad_form(weights, corr)

        # Optimisation objective and constraints
        allocation_objective = cp.Minimize(portfolio_correlation)
        allocation_constraints = [
            cp.sum(weights) == 1
        ]
        if isinstance(self.weight_bounds, tuple):
            allocation_constraints.extend(
                [
                    weights >= self.weight_bounds[0],
                    weights <= min(self.weight_bounds[1], 1)
                ]
            )
        else:
            for inequality in self.weight_bounds:
                allocation_constraints.append(eval(inequality))

            # Add the hard-boundaries for weights.
            allocation_constraints.extend(
                [
                    weights <= 1,
                    weights >= 0
                ]
            )

        # Define and solve the problem
        problem = cp.Problem(
            objective=allocation_objective,
            constraints=allocation_constraints
        )
        problem.solve(warm_start=True)
        if weights.value is None:
            raise ValueError('No optimal set of weights found.')

        self.weights = weights.value
        self.portfolio_risk = risk.value
        self.portfolio_return = portfolio_return.value[0]
