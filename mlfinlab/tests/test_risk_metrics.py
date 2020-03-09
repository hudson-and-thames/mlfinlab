# pylint: disable=missing-module-docstring
import unittest
import os
import pandas as pd
from mlfinlab.portfolio_optimization.risk_metrics import RiskMetrics


class TestRiskMetrics(unittest.TestCase):
    """
    Tests different risk metrics calculation from the RiskMetrics class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """
        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_variance_calculation(self):
        """
        Test the calculation of variance.
        """

        weights = [1] * self.data.shape[1]
        variance = RiskMetrics().calculate_variance(self.data.cov(), weights)
        assert isinstance(variance, float)

    def test_value_at_risk_calculation(self):
        """
        Test the calculation of value at risk.
        """

        test_returns_array = self.data.iloc[:, 0].values
        test_returns_df = self.data.iloc[:, 0]
        value_at_risk_1 = RiskMetrics().calculate_value_at_risk(test_returns_array)
        value_at_risk_2 = RiskMetrics().calculate_value_at_risk(test_returns_df)
        assert isinstance(value_at_risk_1, float)
        assert isinstance(value_at_risk_2, float)

    def test_expected_shortfall_calculation(self):
        """
        Test the calculation of expected shortfall.
        """

        test_returns_array = self.data.iloc[:, 0].values
        test_returns_df = self.data.iloc[:, 0]
        expected_shortfall_1 = RiskMetrics().calculate_expected_shortfall(test_returns_array)
        expected_shortfall_2 = RiskMetrics().calculate_expected_shortfall(test_returns_df)
        assert isinstance(expected_shortfall_1, float)
        assert isinstance(expected_shortfall_2, float)

    def test_conditional_drawdown_calculation(self):
        """
        Test the calculation of conditional drawdown at risk.
        """

        test_returns_array = self.data.iloc[:, 0].values
        test_returns_df = self.data.iloc[:, 0]
        conditional_drawdown_1 = RiskMetrics().calculate_conditional_drawdown_risk(test_returns_array)
        conditional_drawdown_2 = RiskMetrics().calculate_conditional_drawdown_risk(test_returns_df)
        assert isinstance(conditional_drawdown_1, float)
        assert isinstance(conditional_drawdown_2, float)
