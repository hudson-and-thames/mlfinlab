# pylint: disable=missing-module-docstring
import cvxpy as cp
import numpy as np
import pandas as pd
from mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning import CorrelationDrivenNonparametricLearning
from mlfinlab.portfolio_optimization.risk_estimators import RiskEstimators


class CorrelationDrivenNonparametricLearning_DENOISE(CorrelationDrivenNonparametricLearning):
    """
    This class implements the Correlation Driven Nonparametric Learning strategy.
    """

    def calculate_rolling_correlation_coefficient(self, _relative_return):
        """
        Calculates the rolling correlation coefficient for a given relative return and window

        :param _relative_return: (np.array) relative returns of a certain time period specified by the strategy
        :return rolling_corr_coef: (np.array) rolling correlation coefficient over a given window
        """
        # take the log of the relative return
        # first calculate the rolling window the relative return
        # sum the data which returns the log of the window's return
        # take the exp to revert back to the original window's returns
        # calculate the correlation coefficient for the different window's overall returns
        r = RiskEstimators()
        rolling = r.shrinked_covariance(np.exp(np.log(pd.DataFrame(_relative_return)).rolling(self.window).sum()).T)
        rolling = r.cov_to_corr(rolling)
        print(pd.DataFrame(rolling))
        return rolling

    # def calculate_rolling_correlation_coefficient(self, _relative_return):
    #     """
    #     Calculates the rolling correlation coefficient for a given relative return and window
    #
    #     :param _relative_return: (np.array) relative returns of a certain time period specified by the strategy
    #     :return rolling_corr_coef: (np.array) rolling correlation coefficient over a given window
    #     """
    #     # take the log of the relative return
    #     # first calculate the rolling window the relative return
    #     # sum the data which returns the log of the window's return
    #     # take the exp to revert back to the original window's returns
    #     # calculate the correlation coefficient for the different window's overall returns
    #     r = RiskEstimators()
    #     rolling = r.minimum_covariance_determinant(np.exp(np.log(pd.DataFrame(_relative_return)).rolling(self.window).sum()).T)
    #     rolling = r.cov_to_corr(rolling)
    #     print(pd.DataFrame(rolling))
    #     return rolling

def main():
    """

    :return:
    """
    stock_price = pd.read_csv("../../tests/test_data/stock_prices.csv", parse_dates=True, index_col='Date')
    stock_price = stock_price.dropna(axis=1)
    corn_d = CorrelationDrivenNonparametricLearning_DENOISE(window=1, rho=0.1)
    corn_d.allocate(stock_price, resample_by='w')
    print(corn_d.all_weights)
    print(corn_d.portfolio_return)
    corn_d.portfolio_return.plot()


if __name__ == "__main__":
    main()
