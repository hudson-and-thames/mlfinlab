# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimators
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


class MeanVarianceOptimisation:
    # pylint: disable=too-many-instance-attributes
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
                                                 Currently supports: ``mean``, ``exponential``.
        """

        self.weights = list()
        self.asset_names = None
        self.num_assets = None
        self.portfolio_risk = None
        self.portfolio_return = None
        self.portfolio_sharpe_ratio = None
        self.calculate_expected_returns = calculate_expected_returns
        self.returns_estimator = ReturnsEstimators()
        self.risk_estimators = RiskEstimators()
        self.weight_bounds = (0, 1)
        self.risk_free_rate = risk_free_rate

    def allocate(self, asset_names=None, asset_prices=None, expected_asset_returns=None, covariance_matrix=None,
                 solution='inverse_variance', target_return=0.2, target_risk=0.01, risk_aversion=10, weight_bounds=None):
        # pylint: disable=invalid-name, too-many-branches
        """
        Calculate the portfolio asset allocations using the method specified.

        :param asset_names: (list) A list of strings containing the asset names.
        :param asset_prices: (pd.DataFrame) A dataframe of historical asset prices (daily close).
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.DataFrame/numpy matrix) User supplied covariance matrix of asset returns (sigma).
        :param solution: (str) The type of solution/algorithm to use to calculate the weights.
                               Supported solution strings - ``inverse_variance``, ``min_volatility``, ``max_sharpe``,
                               ``efficient_risk``, ``max_return_min_volatility``, ``max_diversification``, ``efficient_return``
                               and ``max_decorrelation``.
        :param target_return: (float) Target return of the portfolio.
        :param target_risk: (float) Target risk of the portfolio.
        :param risk_aversion: (float) Quantifies the risk averse nature of the investor - a higher value means
                                      more risk averse and vice-versa.
        :param weight_bounds: (dict/tuple) Can be either a single tuple of upper and lower bounds
                                           for all portfolio weights or a list of strings with each string representing
                                           an inequality on the weights. For e.g. to bound the weight of the 3rd asset
                                           pass the following weight bounds: ['weights[2] <= 0.3', 'weights[2] >= 0.1'].
        """

        self._error_checks(asset_names, asset_prices, expected_asset_returns, covariance_matrix, solution)

        # Weight bounds
        if weight_bounds is not None:
            self.weight_bounds = weight_bounds

        # Calculate the expected asset returns and covariance matrix if not given by the user
        expected_asset_returns, covariance = self._calculate_estimators(asset_prices,
                                                                        expected_asset_returns,
                                                                        covariance_matrix)

        if solution == 'inverse_variance':
            self._inverse_variance(covariance=covariance, expected_returns=expected_asset_returns)
        elif solution == 'min_volatility':
            self._min_volatility(covariance=covariance,
                                 expected_returns=expected_asset_returns)
        elif solution == 'max_return_min_volatility':
            self._max_return_min_volatility(covariance=covariance,
                                            expected_returns=expected_asset_returns,
                                            risk_aversion=risk_aversion)
        elif solution == 'max_sharpe':
            self._max_sharpe(covariance=covariance,
                             expected_returns=expected_asset_returns)
        elif solution == 'efficient_risk':
            self._min_volatility_for_target_return(covariance=covariance,
                                                   expected_returns=expected_asset_returns,
                                                   target_return=target_return)
        elif solution == 'efficient_return':
            self._max_return_for_target_risk(covariance=covariance,
                                             expected_returns=expected_asset_returns,
                                             target_risk=target_risk)
        elif solution == 'max_diversification':
            self._max_diversification(covariance=covariance,
                                      expected_returns=expected_asset_returns)
        else:
            self._max_decorrelation(covariance=covariance,
                                    expected_returns=expected_asset_returns)

        # Calculate the portfolio sharpe ratio
        self.portfolio_sharpe_ratio = ((self.portfolio_return - self.risk_free_rate) / (self.portfolio_risk ** 0.5))

        # Do some post-processing of the weights
        self._post_process_weights()

    def allocate_custom_objective(self, non_cvxpy_variables, cvxpy_variables, objective_function, constraints=None):
        # pylint: disable=eval-used, exec-used
        """
        Create a portfolio using custom objective and constraints.

        :param non_cvxpy_variables: (dict) A dictionary of variables to be used for providing the required input matrices and
                                           other inputs required by the user. The key of dictionary will be the variable name
                                           while the value can be anything ranging from a numpy matrix, list, dataframe or number.
        :param cvxpy_variables: (list) This is a list of cvxpy specific variables that will be initialised in the format required
                                       by cvxpy. For e.g. ["risk = cp.quad_form(weights, covariance)"] where you are initialising
                                       a variable named "risk" using cvxpy. Note that cvxpy is being imported as "cp", so be sure
                                       to refer to cvxpy as cp.
        :param custom_objective: (str)  A custom objective function. You need to write it in the form
                                        expected by cvxpy. The objective will be a single string, e.g. 'cp.Maximise(
                                        expected_asset_returns)'.
        :param constraints: (list) a list of strings containing the optimisation constraints. For e.g. ['weights >= 0', 'weights <= 1']
        """

        # Initialise the non-cvxpy variables
        locals_ptr = locals()
        for variable_name, variable_value in non_cvxpy_variables.items():
            exec(variable_name + " = None")
            locals_ptr[variable_name] = variable_value

        self.num_assets = locals_ptr['num_assets']
        self.asset_names = list(range(self.num_assets))
        if 'asset_names' in locals_ptr:
            self.asset_names = locals_ptr['asset_names']

        # Optimisation weights
        weights = cp.Variable(self.num_assets)
        weights.value = np.array([1 / self.num_assets] * self.num_assets)

        # Initialise cvxpy specific variables
        for variable in cvxpy_variables:
            exec(variable)

        # Optimisation objective and constraints
        allocation_objective = eval(objective_function)
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

        # Calculate portfolio metrics
        if 'risk' in locals_ptr:
            self.portfolio_risk = locals_ptr['risk'].value
        if 'portfolio_return' in locals_ptr:
            self.portfolio_return = locals_ptr['portfolio_return'].value

        # Do some post-processing of the weights
        self._post_process_weights()

    def get_portfolio_metrics(self):
        """
        Prints the portfolio metrics - return, risk and Sharpe Ratio.
        """

        print("Portfolio Return = %s" % self.portfolio_return)
        print("Portfolio Risk = %s" % self.portfolio_risk)
        print("Portfolio Sharpe Ratio = %s" % self.portfolio_sharpe_ratio)

    def plot_efficient_frontier(self, covariance, expected_asset_returns, min_return=0, max_return=0.4,
                                risk_free_rate=0.05):
        # pylint: disable=broad-except
        """
        Plot the Markowitz efficient frontier.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
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
                self.allocate(covariance_matrix=covariance,
                              expected_asset_returns=expected_returns,
                              solution='efficient_risk',
                              target_return=portfolio_return)
                volatilities.append(self.portfolio_risk ** 0.5)
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

    def _error_checks(self, asset_names, asset_prices, expected_asset_returns, covariance_matrix, solution=None):
        """
        Some initial error checks on the inputs.

        :param asset_names: (list) A list of strings containing the asset names.
        :param asset_prices: (pd.DataFrame) A dataframe of historical asset prices (daily close).
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.DataFrame/numpy matrix) User supplied covariance matrix of asset returns (sigma).
        :param solution: (str) The type of solution/algorithm to use to calculate the weights.
                               Currently supported solution strings - inverse_variance, min_volatility, max_sharpe,
                               efficient_risk, max_return_min_volatility, max_diversification, efficient_return
                               and max_decorrelation.
        """

        if asset_prices is None and (expected_asset_returns is None or covariance_matrix is None):
            raise ValueError("You need to supply either raw prices or expected returns "
                             "and a covariance matrix of asset returns")

        if asset_prices is not None:
            if not isinstance(asset_prices, pd.DataFrame):
                raise ValueError("Asset prices matrix must be a dataframe")
            if not isinstance(asset_prices.index, pd.DatetimeIndex):
                raise ValueError("Asset prices dataframe must be indexed by date.")

        if solution is not None and solution not in {"inverse_variance", "min_volatility", "max_sharpe", "efficient_risk",
                                                     "max_return_min_volatility", "max_diversification", "efficient_return",
                                                     "max_decorrelation"}:
            raise ValueError("Unknown solution string specified. Supported solutions - "
                             "inverse_variance, min_volatility, max_sharpe, efficient_risk"
                             "max_return_min_volatility, max_diversification, efficient_return and max_decorrelation")

        if asset_names is None:
            if asset_prices is not None:
                asset_names = asset_prices.columns
            elif covariance_matrix is not None and isinstance(covariance_matrix, pd.DataFrame):
                asset_names = covariance_matrix.columns
            else:
                raise ValueError("Please provide a list of asset names")
        self.asset_names = asset_names
        self.num_assets = len(asset_names)

    def _calculate_estimators(self, asset_prices, expected_asset_returns, covariance_matrix):
        """
        Calculate the expected returns and covariance matrix of assets in the portfolio.

        :param asset_prices: (pd.DataFrame) A dataframe of historical asset prices (daily close).
        :param expected_asset_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param covariance_matrix: (pd.DataFrame/numpy matrix) User supplied covariance matrix of asset returns (sigma).
        :return: (np.array, pd.DataFrame) Expected asset returns and covariance matrix.
        """

        # Calculate the expected returns if the user does not supply any returns
        if expected_asset_returns is None:
            if self.calculate_expected_returns == "mean":
                expected_asset_returns = self.returns_estimator.calculate_mean_historical_returns(
                    asset_prices=asset_prices)
            elif self.calculate_expected_returns == "exponential":
                expected_asset_returns = self.returns_estimator.calculate_exponential_historical_returns(
                    asset_prices=asset_prices)
            else:
                raise ValueError("Unknown returns specified. Supported returns - mean, exponential")
        expected_asset_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))

        # Calculate covariance of returns or use the user specified covariance matrix
        if covariance_matrix is None:
            returns = self.returns_estimator.calculate_returns(asset_prices=asset_prices)
            covariance_matrix = returns.cov()
        cov = pd.DataFrame(covariance_matrix, index=self.asset_names, columns=self.asset_names)

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

        self.weights = pd.DataFrame(self.weights)
        self.weights.index = self.asset_names
        self.weights = self.weights.T

    def _inverse_variance(self, covariance, expected_returns):
        """
        Calculate weights using inverse-variance allocation.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        """

        ivp = 1. / np.diag(covariance)
        ivp /= ivp.sum()
        self.weights = ivp
        self.portfolio_risk = np.dot(self.weights, np.dot(covariance.values, self.weights.T))
        self.portfolio_return = np.dot(self.weights, expected_returns)[0]

    def _min_volatility(self, covariance, expected_returns):
        # pylint: disable=eval-used
        """
        Compute minimum volatility portfolio allocation.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        """

        weights = cp.Variable(self.num_assets)
        weights.value = np.array([1 / self.num_assets] * self.num_assets)
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

    def _max_return_min_volatility(self, covariance, expected_returns, risk_aversion):
        # pylint: disable=eval-used
        """
        Calculate maximum return-minimum volatility portfolio allocation.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param risk_aversion: (float) Quantifies the risk-averse nature of the investor - a higher value means
                           more risk averse and vice-versa.
        """

        weights = cp.Variable(self.num_assets)
        weights.value = np.array([1 / self.num_assets] * self.num_assets)
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

    def _max_sharpe(self, covariance, expected_returns):
        # pylint: disable=invalid-name, eval-used
        """
        Compute maximum Sharpe portfolio allocation.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        """

        y = cp.Variable(self.num_assets)
        y.value = np.array([1 / self.num_assets] * self.num_assets)
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

    def _min_volatility_for_target_return(self, covariance, expected_returns, target_return):
        # pylint: disable=eval-used
        """
        Calculate minimum volatility portfolio for a given target return.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param target_return: (float) Target return of the portfolio.
        """

        weights = cp.Variable(self.num_assets)
        weights.value = np.array([1 / self.num_assets] * self.num_assets)
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

    def _max_return_for_target_risk(self, covariance, expected_returns, target_risk):
        # pylint: disable=eval-used
        """
        Calculate maximum return for a given target volatility/risk.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        :param target_risk: (float) Target risk of the portfolio.
        """

        weights = cp.Variable(self.num_assets)
        weights.value = np.array([1 / self.num_assets] * self.num_assets)
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

    def _max_diversification(self, covariance, expected_returns):
        """
        Calculate the maximum diversified portfolio.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        """

        self._max_decorrelation(covariance, expected_returns)

        # Divide weights by individual asset volatilities
        self.weights /= np.diag(covariance)

        # Standardize weights
        self.weights /= np.sum(self.weights)

        portfolio_return = np.dot(expected_returns.T, self.weights)[0]
        risk = np.dot(self.weights, np.dot(covariance, self.weights.T))

        self.portfolio_risk = risk
        self.portfolio_return = portfolio_return

    def _max_decorrelation(self, covariance, expected_returns):
        # pylint: disable=eval-used
        """
        Calculate the maximum decorrelated portfolio.

        :param covariance: (pd.DataFrame) Covariance dataframe of asset returns.
        :param expected_returns: (list/np.array/pd.dataframe) A list of mean stock returns (mu).
        """

        weights = cp.Variable(self.num_assets)
        weights.value = np.array([1 / self.num_assets] * self.num_assets)
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
