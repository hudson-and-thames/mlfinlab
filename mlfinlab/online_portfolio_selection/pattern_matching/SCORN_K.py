# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning_k import CorrelationDrivenNonparametricLearningK
from mlfinlab.online_portfolio_selection.pattern_matching.symmetric_correlation_driven_nonparametric_learning import SymmetricCorrelationDrivenNonparametricLearning


class SCORN_K(CorrelationDrivenNonparametricLearningK):
    """
    This class implements the Symmetirc Correlation Driven Nonparametric Learning - top k experts strategy.
    """

    def _generate_experts(self):
        """
        Generates n experts for SymmetricCorrelationDrivenNonparametricLearning-K strategy

        :return:
        """
        self.expert_params = np.zeros((self.number_of_experts, 2))
        pointer = 0
        for _window in self.window_values:
            for _rho in self.rho_values:
                self.expert_params[pointer] = [_window, _rho]
                pointer += 1

        for exp in range(self.number_of_experts):
            param = self.expert_params[exp]
            self.experts.append(SymmetricCorrelationDrivenNonparametricLearning(int(param[0]), param[1]))


def main():
    """
    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    scorn_k = SCORN_K(k=3, window_values=[2,3,4], rho_values=[0.4,0.6,0.8])
    scorn_k.allocate(stock_price, resample_by='m')
    print(scorn_k.all_weights)
    print(scorn_k.portfolio_return)
    scorn_k.portfolio_return.plot()


if __name__ == "__main__":
    main()
