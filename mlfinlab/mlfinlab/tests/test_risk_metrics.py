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

        test_returns = self.data.iloc[:, 0].values
        value_at_risk = RiskMetrics().calculate_value_at_risk(test_returns)
        assert isinstance(value_at_risk, float)

    def test_expected_shortfall_calculation(self):
        """
        Test the calculation of expected shortfall.
        """

        test_returns = self.data.iloc[:, 0].values
        expected_shortfall = RiskMetrics().calculate_expected_shortfall(test_returns)
        assert isinstance(expected_shortfall, float)

    def test_conditional_drawdown_calculation(self):
        """
        Test the calculation of conditional drawdown at risk.
        """

        test_returns = self.data.iloc[:, 0].values
        conditional_drawdown = RiskMetrics().calculate_conditional_drawdown_risk(test_returns)
        assert isinstance(conditional_drawdown, float)

    def test_value_at_risk_for_dataframe(self):
        """
        Test the calculation of value at risk.
        """

        test_returns = pd.DataFrame(self.data.iloc[:, 0])
        value_at_risk = RiskMetrics().calculate_value_at_risk(test_returns)
        assert isinstance(value_at_risk, float)

    def test_expected_shortfall_for_dataframe(self):
        """
        Test the calculation of expected shortfall.
        """

        test_returns = pd.DataFrame(self.data.iloc[:, 0])
        expected_shortfall = RiskMetrics().calculate_expected_shortfall(test_returns)
        assert isinstance(expected_shortfall, float)

    def test_conditional_drawdown_for_dataframe(self):
        """
        Test the calculation of conditional drawdown at risk.
        """

        test_returns = pd.DataFrame(self.data.iloc[:, 0])
        conditional_drawdown = RiskMetrics().calculate_conditional_drawdown_risk(test_returns)
        assert isinstance(conditional_drawdown, float)
