# pylint: disable=missing-module-docstring
import numpy as np
from mlfinlab.online_portfolio_selection.cornk import CORNK
from mlfinlab.online_portfolio_selection.scorn import SCORN


class SCORNK(CORNK):
    """
    This class implements the Symmetric Correlation Driven Nonparametric Learning - K strategy. It
    is reproduced with modification from the following paper:
    `Wang, Y., & Wang, D. (2019). Market Symmetry and Its Application to Pattern-Matching-Based
    Portfolio Selection. The Journal of Financial Data Science, 1(2), 78–92.
    <https://jfds.pm-research.com/content/1/2/78>`_

    Symmetric Correlation Driven Nonparametric Learning - K formulates a number of SCORN experts and
    tracks the experts performance over time. Each period, the strategy decides to allocate capital
    to the top-k experts until the previous time period. This strategy takes an ensemble approach to
    the top performing experts.
    """

    def _generate_experts(self):
        """
        Generates window * rho experts from window of [1, w] and rho of [0, (rho - 1) / rho].
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

        for exp in range(self.number_of_experts):
            param = self.expert_params[exp]
            self.experts.append(SCORN(int(param[0]), param[1]))
