# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.universal_portfolio import UniversalPortfolio
from mlfinlab.online_portfolio_selection.pattern_matching.functional_correlation_driven_nonparametric_learning import \
    FunctionalCorrelationDrivenNonparametricLearning


class FunctionalCorrelationDrivenNonparametricLearningK(UniversalPortfolio):
    """
    This class implements the Functional Correlation Driven Nonparametric Learning - K strategy. It
    is reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Functional Correlation Driven Nonparametric Learning - K formulates a number of FCORN experts and
    tracks the experts performance over time. Each period, the strategy decides to allocate capital
    to the top-k experts until the previous time period. This strategy takes an ensemble approach to
    the top performing experts.
    """

    def __init__(self, window, rho, lambd, k):
        """
        Initializes Functional Correlation Driven Nonparametric Learning - K with the given number
        of window, rho, lambda, and k experts.

        :param window: (int) Number of windows to look back for similarity sets. (1, 2, ..., w).
        :param rho: (float) Number of rho values for threshold. (0, 1/rho, ..., (rho-1)/rho).
        :param lambd: (float) Scale factor for sigmoid function. (1, 10, 100, 10 ** (lambd-1))
        :param k: (int) Number of top-k experts.
        """
        self.window = window
        self.rho = rho
        self.lambd = lambd
        self.k = k
        self.number_of_experts = self.window * self.rho * self.lambd
        super().__init__(number_of_experts=self.number_of_experts, weighted='top-k', k=self.k)

    def _initialize(self, asset_prices, weights, resample_by):
        """
        Initializes the important variables for the object.

        :param asset_prices: (pd.DataFrame) Historical asset prices.
        :param weights: (list/np.array/pd.Dataframe) Initial weights set by the user.
        :param resample_by: (str) Specifies how to resample the prices.
        """
        # Check that window value is an integer.
        if not isinstance(self.window, int):
            raise ValueError("Window value must be an integer.")

        # Check that rho value is an integer.
        if not isinstance(self.rho, int):
            raise ValueError("Rho value must be an integer.")

        # Check that lambd value is an integer.
        if not isinstance(self.lambd, int):
            raise ValueError("Lambd value must be an integer.")

        # Check that window value is at least 1.
        if self.window < 1:
            raise ValueError("Window value must be greater than or equal to 1.")

        # Check that rho value is at least 1.
        if self.rho < 1:
            raise ValueError("Rho value must be greater than or equal to 1.")

        # Check that lambd value is at least 1.
        if self.lambd < 1:
            raise ValueError("Lambd value must be greater than or equal to 0.")

        # Check that k value is at least the number of experts.
        if self.k > self.number_of_experts:
            raise ValueError("K must be less than window * rho * lambd.")

        super(FunctionalCorrelationDrivenNonparametricLearningK, self)._initialize(asset_prices,
                                                                                   weights,
                                                                                   resample_by)

    def _generate_experts(self):
        """
        Generates window * rho experts from window of 1 to w, rho of 0 to (rho-1)/rho, and
        lambd of 1 to 10 ** (lambd-1).
        """
        # Initialize expert parameters.
        self.expert_params = np.zeros((self.number_of_experts, 3))
        # Pointer to iterate through parameter locations.
        pointer = 0
        # Window from 1 to self.window.
        for n_window in range(self.window):
            # Rho from 0 to (rho - 1)/rho.
            for n_rho in range(self.rho):
                for n_lambd in range(self.lambd):
                    # Assign experts with parameters (n_window + 1, n_rho/rho, 10 ** n_lambd).
                    self.expert_params[pointer] = [n_window + 1, n_rho / self.rho, 10 ** n_lambd]
                    pointer += 1

        for exp in range(self.number_of_experts):
            param = self.expert_params[exp]
            self.experts.append(
                FunctionalCorrelationDrivenNonparametricLearning(int(param[0]), param[1], param[2]))
