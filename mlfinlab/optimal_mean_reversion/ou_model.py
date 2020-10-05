# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

# pylint: disable=missing-module-docstring, invalid-name
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd


class OrnsteinUhlenbeck:
    """
    This class implements the algorithm for solving the optimal stopping problem in
    markets with mean-reverting tendencies based on the Ornstein-Uhlenbeck model
    mentioned in the following publication:'Tim Leung and Xin Li Optimal Mean
    reversion Trading: Mathematical Analysis and Practical Applications(November 26, 2015)'
    <https://www.amazon.com/Optimal-Mean-Reversion-Trading-Mathematical/dp/9814725919>`_

    Constructing a portfolio with mean-reverting properties is usually attempted by
    simultaneously taking a position in two highly correlated or co-moving assets and is
    labeled as "pairs trading". One of the most important problems faced by investors is
    to determine when to open and close a position.

    To find the liquidation and entry price levels we formulate an optimal double-stopping
    problem that gives the optimal entry and exit level rules. Also, a stop-loss
    constraint is incorporated into this trading problem and solutions are also provided
    by this module.
    """

    def __init__(self):

        pass

    def fit(self, data, data_frequency, discount_rate, transaction_cost, start=None, end=None,
            stop_loss=None):
        """
        Fits the Ornstein-Uhlenbeck model to given data and assigns the discount rates,
        transaction costs and stop-loss level for further exit or entry-level calculation.

        :param data: (np.array/pd.DataFrame) An array with time series of portfolio prices / An array with
            time series of of two assets prices. The dimensions should be either nx1 or nx2.
        :param data_frequency: (str) Data frequency ["D" - daily, "M" - monthly, "Y" - yearly].
        :param discount_rate: (float/tuple) A discount rate either for both entry and exit time
            or a list/tuple of discount rates with exit rate and entry rate in respective order.
        :param transaction_cost: (float/tuple) A transaction cost either for both entry and exit time
            or a list/tuple of transaction costs with exit cost and entry cost in respective order.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        :param stop_loss: (float/int) A stop-loss level - the position is assumed to be closed
            immediately upon reaching this pre-defined price level.
        """

        pass

    def _fit_data(self, start, end):
        """
        Allocates all the training data parameters
        and fits the OU model to this data.

        Helper function used in self.fit().

        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        :return: (np.array) An array of prices to construct a portfolio from.
        """

        pass

    @staticmethod
    def _fit_rate_cost(input_data):
        """
        Sets the value for cost and rate parameters.

        Helper function used in self.fit().

        :param input_data: (float/tuple) Input for cost or rate.
        :return: A tuple of two elements with allocated data for cost/rate
        """

        pass

    def _fit_delta(self, data_frequency):
        """
        Sets the value of the delta-t parameter,
        depending on data frequency input.

        Helper function used in self.fit().

        :param data_frequency: (str) Data frequency
            ["D" - daily, "M" - monthly, "Y" - yearly].
        """

        pass

    def fit_to_portfolio(self, data=None, start=None, end=None):
        """
        Fits the Ornstein-Uhlenbeck model to time series
        for portfolio prices.

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """

        pass

    @staticmethod
    def portfolio_from_prices(prices, b_variable):
        """
        Constructs a portfolio based on two given asset prices
        and the relative amount of investment for one of them.

        :param prices: (np.array) An array of prices of the two assets
            used to create a portfolio.
        :param b_variable: (float) A coefficient representing the investment.
            into the second asset, investing into the first one equals one.
        :return: (np.array) Portfolio prices. (p. 11)
        """

        pass

    def half_life(self):
        """
        Returns the half-life of the fitted OU process. Half-life stands for the average time that
        it takes for the process to revert to its long term mean on a half of its initial deviation.

        :return: (float) Half-life of the fitted OU process
        """

        pass

    def fit_to_assets(self, data=None, start=None, end=None):
        """
        Creates the optimal portfolio in terms of Ornstein-Uhlenbeck model
        from two given time series for asset prices and fits the values
        of the model's parameters. (p.13)

        :param data: (np.array) All given prices of two assets to construct a portfolio from.
        :param start: (Datetime) A date from which you want your training data to start.
        :param end: (Datetime) A date at which you want your training data to end.
        """

        pass

    def ou_model_simulation(self, n, theta_given=None, mu_given=None, sigma_given=None, delta_t_given=None):
        """
        Simulates values of an OU process with given parameters or parameters fitted to our data

        :param n: (int) Number of simulated values.
        :param theta_given: (float) Long-term mean.
        :param mu_given: (float) Mean revesion speed.
        :param sigma_given: (float) The amplitude of randomness in the system.
        :param delta_t_given: (float) Delta between observations, calculated in years.
        :return: (np.array) simulated portfolio prices.
        """

        pass

    def optimal_coefficients(self, portfolio):
        """
        Finds the optimal Ornstein-Uhlenbeck model coefficients depending
        on the portfolio prices time series given.(p.13)

        :param portfolio: (np.array) Portfolio prices.
        :return: (tuple) Optimal parameters (theta, mu, sigma_square, max_LL)
        """

        pass


    def check_fit(self):
        """
        Shows the parameters of the OU model fitted to the data and parameters achieved
        from the simulated process with the fitted coefficients for further assessment. The main
        incentive is to have simulated max log-likelihood (mll) function close to a fitted one.
        If the simulated mll is much greater it means that data provided to
        the model is not good enough to be modeled by an OU process.

        Note: The check can occasionally return numbers significantly different from the fitted mll
        due to the random nature of the simulated process and the possibility of outliers. It is advised
        to perform the check multiple times and if it's consistently showing very different values of mll

        you might suspect the unsuitability of your data.
        :return: (pd.Dataframe) Values for fitted and simulated OU model parameters
        """

        pass

    @staticmethod
    def _compute_log_likelihood(params, *args):
        """
        Computes the average Log Likelihood. (p.13)

        :param params: (tuple) A tuple of three elements representing theta, mu and sigma_squared.
        :param args: (tuple) All other values that to be passed to self._compute_log_likelihood()
        :return: (float) The average log likelihood from given parameters.
        """

        pass

    def description(self):
        """
        Returns all the general parameters of the model, training interval timestamps if provided, the goodness of fit,
        allocated trading costs and discount rates, stop-loss level, beta, which stands for the optimal ratio between
        two assets in the created portfolio, and optimal levels calculated. If the stop-loss level was given optimal
        levels that account for stop-loss would be added to the list.

        :return: (pd.Series) Summary data for all model parameters and optimal levels.
        """

    def plot_levels(self, data, stop_loss=False):
        """
        Plots the found optimal exit and entry levels on the graph
        alongside with the given data.

        :param data: (np.array/pd.DataFrame) Time series of portfolio prices /
            time series of of two assets prices of size (n x 2).
        :param stop_loss: (bool) A flag whether to take stop-loss level into account.
            when showcasing the results.
        :return: (plt.Figure) Figure with optimal exit and entry levels.
        """

        pass

    def _F(self, price, rate):
        """
        Calculates helper function to further define the exit/enter level. (p.18)

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :return: (float) Value of F function.
        """

        pass

    def _F_derivative(self, price, rate, h=1e-4):
        """
        Calculates a derivative with respect to price of a helper function
        to further define the exit/enter level.

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Value of F derivative function.
        """

        pass

    def _G(self, price, rate):
        """
        Calculates helper function to further define the exit/enter level. (p.18)

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :return: (float) Value of G function.
        """

        pass

    def _G_derivative(self, price, rate, h=1e-4):
        """
        Calculate a derivative with respect to price to a helper function to
        further define the exit/enter level.

        :param price: (float) Portfolio price.
        :param rate: (float) Discounting rate.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Value of G derivative function.
        """

        pass

    def V(self, price):
        """
        Calculates the expected discounted value of liquidation of the position. (p.23)

        :param price: (float) Portfolio value.
        :return: (float) Expected discounted liquidation value.
        """

        pass

    def _V_derivative(self, price, h=1e-4):
        """
        Calculates the derivative of the expected discounted value of
        liquidation of the position.

        :param price: (float) Portfolio value.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Value of V derivative function.
        """

        pass

    def optimal_liquidation_level(self):
        """
        Calculates the optimal liquidation portfolio level. (p.23)

        :return: (float) Optimal liquidation portfolio level.
        """

        pass

    def optimal_entry_level(self):
        """
        Calculates the optimal entry portfolio level. (p.27)

        :return: (float) Optimal entry portfolio level.
        """

        pass

    def _C(self):
        """
        Calculates helper function to further define the exit/enter
        level with a stop-loss level. (p.31)

        :return: (float) Value of C function.
        """

        pass

    def _D(self):
        """
        Calculates helper function to further define the exit/enter level
        with a stop-loss level. (p.31)

        :return: (float) Value of D function.
        """

        pass

    def V_sl(self, price):
        """
        Calculates the expected discounted value of liquidation of the position
        considering the stop-loss level. (p. 31)

        :param price: (float) Portfolio value.
        :return: (float) Expected discounted value of liquidating the position
            considering the stop-loss level.
        """

        pass

    def _V_sl_derivative(self, price, h=1e-4):
        """
        Calculates the derivative of the expected discounted value of liquidation
        of the position considering the stop-loss level.

        :param price: (float) Portfolio value.
        :param h: (float) Delta step to use to calculate derivative.
        :return: (float) Expected discounted value of liquidating the position
            considering the stop-loss level.
        """

        pass

    def optimal_liquidation_level_stop_loss(self):
        """
        Calculates the optimal liquidation portfolio level considering the stop-loss level. (p.31)

        :return: (float) Optimal liquidation portfolio level considering the stop-loss.
        """

        pass

    def optimal_entry_interval_stop_loss(self):
        """
        Calculates the optimal entry portfolio interval considering the stop-loss level. (p.35)

        :return: (tuple) Optimal entry portfolio interval considering the stop-loss.
        """

        pass

    def _parameter_check(self):
        """
        Checks if fitted parameters satisfy the necessary condition to calculate
        optimal entry level accounting for stop-loss. (p.34)

        Condition:
        sup_x{V_L(x) - x - cb} > 0

        :return: (bool) The result of the check.
        """

        pass

