# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.universal_portfolio import UniversalPortfolio
from mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning import \
    CorrelationDrivenNonparametricLearning


class CorrelationDrivenNonparametricLearningK(UniversalPortfolio):
    """
    This class implements the Correlation Driven Nonparametric Learning - K strategy. It is
    reproduced with modification from the following paper:
    `Li, B., Hoi, S.C., & Gopalkrishnan, V. (2011). CORN: Correlation-driven nonparametric
    learning approach for portfolio selection. ACM TIST, 2,
    21:1-21:29.<https://dl.acm.org/doi/abs/10.1145/1961189.1961193>`_

    Correlation Driven Nonparametric Learning - K formulates a number of experts and tracks the
    experts performance over time. Each period, the strategy decides to allocate capital to
    the top-k experts until the previous time period. This strategy takes an ensemble approach to
    the top performing experts.
    """

    def __init__(self, window, rho, k):
        """
        Initializes Correlation Driven Nonparametric Learning - K with the given number of
        windows, rho values, and k experts.

        :param window: (int) Number of windows to look back for similarity sets. (1, 2, ..., w).
        :param rho: (float) Number of rho values for threshold. (0, 1/rho, ..., (rho-1)/rho).
        :param k: (int) Number of top-k experts.
        """
        self.window = window
        self.rho = rho
        self.k = k
        self.number_of_experts = self.window * self.rho
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

        # Check that k value is an integer.
        if not isinstance(self.k, int):
            raise ValueError("K value must be an integer.")

        # Check that window value is at least 1.
        if self.window < 1:
            raise ValueError("Window value must be greater than or equal to 1.")

        # Check that rho value is at least 1.
        if self.rho < 1:
            raise ValueError("Rho value must be greater than or equal to 1.")

        # Check that k value is at least 1.
        if self.k < 1:
            raise ValueError("K value must be greater than or equal to 1.")

        # Check that k value is less than window * rho.
        if self.k > self.number_of_experts:
            raise ValueError("K must be less than or equal to window * rho.")

        super(CorrelationDrivenNonparametricLearningK, self)._initialize(asset_prices,
                                                                         weights,
                                                                         resample_by)

    def _generate_experts(self):
        """
        Generates window * rho experts from window of 1 to w and rho of 0 to (rho-1)/rho.
        """
        # Initialize expert parameters.
        self.expert_params = np.zeros((self.number_of_experts, 2))

        # Pointer to iterate through parameter locations.
        pointer = 0

        # Window from 1 to self.window.
        for n_window in range(self.window):
            # Rho from 0 to (rho - 1)/rho.
            for n_rho in range(self.rho):
                # Assign experts with parameters (n_window + 1, n_rho/rho).
                self.expert_params[pointer] = [n_window + 1, n_rho/self.rho]
                # Next pointer.
                pointer += 1
        # Assign parameters.
        for exp in range(self.number_of_experts):
            param = self.expert_params[exp]
            self.experts.append(CorrelationDrivenNonparametricLearning(int(param[0]), param[1]))