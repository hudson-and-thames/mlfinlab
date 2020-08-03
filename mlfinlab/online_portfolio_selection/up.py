# pylint: disable=missing-module-docstring
# pylint: disable=too-many-instance-attributes
import sys
import numpy as np
from mlfinlab.online_portfolio_selection.base import OLPS
from mlfinlab.online_portfolio_selection.crp import CRP


class UP(OLPS):
    """
    This class implements the Universal Portfolio strategy. It is reproduced with
    modification from the following paper:
    `Cover, T.M. (1991), Universal Portfolios. Mathematical Finance, 1: 1-29.
    <http://www-isl.stanford.edu/~cover/papers/portfolios_side_info.pdf>`_

    Universal Portfolio acts as a fund of funds, generating a number of experts with unique
    strategies. Cover's original universal portfolio integrates over the total simplex domain,
    but because it is not possible for us to calculate all possibilties, we generate a random
    distribution of points.

    The allocation method to each experts can be changed. If no
    allocation method is given, Universal Portfolio will not rebalance among the experts. Other
    allocation methods include uniform allocation among experts and top-k experts, which allocate
    capital based on the top-k performing experts until the last period.
    """
    def __init__(self, number_of_experts, weighted='hist_performance', k=1):
        """
        Initializes Universal Portfolio with the given number of experts, method of capital
        allocation to each experts, and k-value for Top-K experts.

        :param number_of_experts: (int) Number of total experts.
        :param weighted: (str) Capital allocation method. 'hist_performance': Historical Performance,
                               'uniform': Uniform Weights, 'top-k': Top-K experts.
        :param k: (int) Number of experts to choose your portfolio. Only necessary if weighted is
                        'top-k'. Typically lower values of k are more optimal for higher returns.
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

    def _calculate_weights_on_experts(self):
        """
        Calculates the weight allocation on each experts.
        'hist_performance': Historical Performance.
        'uniform': Uniform Weights.
        'top-k': Top-K experts.
        """

        pass

    def recalculate_k(self, k):
        """
        Calculates the existing strategy with a different k value. The user does not have to
        rerun the entire strategy, but can simply recalculate with another k parameter.

        :param k: (int) Number of new top-k experts.
        """

        pass

    def _uniform_experts(self):
        """
        Returns a uniform weight of experts.

        :return: (np.array) Uniform weights (1/n, 1/n, 1/n ...).
        """

        pass

    def _calculate_all_weights(self):
        # pylint: disable=unsubscriptable-object
        """
        Calculate portfolio's overall weights and final predicted weights with information from
        each expert's weights.
        """


        pass

    def _generate_experts(self):
        """
        Generate experts with the specified parameter.
        """

        pass

    def _generate_simplex(self, number_of_experts, number_of_assets):
        """
        Generate uniform points on a simplex domain.
        `<https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex>`_

        :param number_of_experts: (int) Number of experts.
        :param number_of_assets: (int) Number of assets.
        """

        pass

    def _print_progress(self, iteration, prefix='', suffix='', decimals=1, bar_length=50):
        # pylint: disable=expression-not-assigned
        """
        Calls in a loop to create a terminal progress bar.
        `<https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a>`_

        :param iteration: (int) Current iteration.
        :param prefix: (str) Prefix string.
        :param suffix: (str) Suffix string.
        :param decimals: (int) Positive number of decimals in percent completed.
        :param bar_length: (int) Character length of the bar.
        """

        pass
