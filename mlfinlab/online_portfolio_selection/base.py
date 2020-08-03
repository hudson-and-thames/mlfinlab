# pylint: disable=missing-module-docstring
import sys
import pandas as pd
import numpy as np
import cvxpy as cp


class OLPS:
    """
    Online Portfolio Selection is an algorithmic trading strategy that sequentially allocates
    capital among a group of assets to maximize the final returns of the investment.

    Traditional theories for portfolio selection, such as Markowitz’s Modern Portfolio Theory,
    optimize the balance between the portfolio's risks and returns. However, OLPS is founded on
    the capital growth theory, which solely focuses on maximizing the returns of the current
    portfolio.

    Through these walkthroughs of different portfolio selection strategies, we hope to introduce
    a set of different selection tools available for everyone. Most of the works will be based on
    Dr. Bin Li and Dr. Steven Hoi’s book, Online Portfolio Selection: Principles and Algorithms,
    and further recent papers will be implemented to assist the development and understanding of
    these unique portfolio selection strategies.

    OLPS is the parent class for all resulting Online Portfolio Selection Strategies.

    This class broadly defines all variables and allocates a set of weights for a certain strategy.

    Upon weights allocation the possible outputs are:

    - ``self.weights`` (np.array) Final portfolio weights prediction.

    - ``self.all_weights`` (pd.DataFrame) Portfolio weights for the time period.

    - ``self.asset_name`` (list) Name of assets.

    - ``self.number_of_assets`` (int) Number of assets.

    - ``self.time`` (datetime) Time index of the given data.

    - ``self.length_of_time`` (int) Number of time periods.

    - ``self.relative_return`` (np.array) Relative returns of the assets.

    - ``self.portfolio_return`` (pd.DataFrame) Cumulative portfolio returns over time.

    - ``self.asset_prices`` (pd.DataFrame) Historical asset prices (daily close).
    """

    def __init__(self):

        pass

    def allocate(self, asset_prices, weights=None, resample_by=None, verbose=False):
        """
        Allocates weight according to a set of update rules.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user. If None, weights
                                                     will default to uniform weights.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        :param verbose: (bool) Prints progress bar if true.
        """

        pass

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices. 'D' for Day, 'W' for Week,
                                 'M' for Month. The inputs are based on pandas' resample method.
        """

        pass

    def _run(self, weights, verbose):
        """
        Runs the algorithm by iterating through the given data.

        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        :param verbose: (bool) Prints progress bar if true.
        """

        pass

    def _first_weight(self, weights):
        """
        Returns the first weight of the given portfolio. If the first weight is not given, initialize weights to
        uniform weights.

        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :return: (np.array) First portfolio weight.
        """

        pass

    def _update_weight(self, time):
        """
        Predicts the next time's portfolio weight.

        :param time: (int) Current time period.
        :return: (np.array) Identical weights are returned for OLPS.
        """

        pass

    def _calculate_portfolio_returns(self, all_weights, relative_return):
        """
        Calculates cumulative portfolio returns.

        :param all_weights: (np.array) Portfolio weights for the time period.
        :param relative_return: (np.array) Relative returns of the assets.
        """

        pass

    def _conversion(self):
        """
        Converts the given np.array to pd.Dataframe.
        """

        pass

    def _optimize(self, optimize_array, solver=cp.SCS):
        """
        Calculates weights that maximize returns over the given array.

        :param optimize_array: (np.array) Relative returns of the assets for a given time period.
        :param solver: (cp.solver) Solver for cvxpy
        :return: (np.array) Weights that maximize the returns for the given array.
        """


        pass

    def _round_weights(self, threshold=1e-6):
        """
        Drops weights that are below a certain threshold.

        :param threshold: (float) Drop all values below this threshold.
        """

        pass

    def _uniform_weight(self):
        """
        Returns a uniform weight of assets.

        :return: (np.array) Uniform weights (1/n, 1/n, 1/n ...).
        """

        pass

    def _print_progress(self, iteration, prefix='', suffix='', decimals=1, bar_length=50):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

        :param iteration: (int) Current iteration.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """

        pass

    @staticmethod
    def _normalize(weights):
        """
        Normalize sum of weights to one.

        :param weights: (np.array) Pre-processed weights that have not been normalized yet.
        :return: (np.array) Adjusted weights that sum to 1.
        """

        pass

    @staticmethod
    def _calculate_relative_return(asset_prices):
        """
        Calculates the relative return of a given price data.

        :param asset_prices: (pd.DataFrame) Dataframe of historical asset prices.
        :return: (np.array) Relative returns of the assets.
        """

        pass

    @staticmethod
    def _check_asset(asset_prices, weights):
        """
        Checks if the given input values are valid.

        :param asset_prices: (pd.DataFrame) Dataframe of historical asset prices.
        :param weights: (list/np.array/pd.DataFrame) Initial weights set by the user.
        """

        pass

    @staticmethod
    def _simplex_projection(weight):
        """
        Calculates the simplex projection of weights.
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

        :param weight: (np.array) Weight to be projected onto the simplex domain.
        :return: (np.array) Simplex projection of the original weight.
        """

        pass
