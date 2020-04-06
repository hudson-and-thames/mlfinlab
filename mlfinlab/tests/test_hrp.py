"""
Tests the Hierarchical Risk Parity (HRP) algorithm.
"""

import unittest
import os
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.returns_estimators import ReturnsEstimation


class TestHRP(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """
    Tests different functions of the HRP algorithm class.
    """

    def setUp(self):
        """
        Set the file path for the tick data csv.
        """

        project_path = os.path.dirname(__file__)
        data_path = project_path + '/test_data/stock_prices.csv'
        self.data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

    def test_hrp(self):
        """
        Test the weights calculated by the HRP algorithm - if all the weights are positive and
        their sum is equal to 1.
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_hrp_with_shrinkage(self):
        """
        Test the weights calculated by HRP algorithm with covariance shrinkage.
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, use_shrinkage=True, asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_dendrogram_plot(self):
        """
        Test if dendrogram plot object is correctly rendered.
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, use_shrinkage=True, asset_names=self.data.columns)
        dendrogram = hrp.plot_clusters(assets=self.data.columns)
        assert dendrogram.get('icoord')
        assert dendrogram.get('dcoord')
        assert dendrogram.get('ivl')
        assert dendrogram.get('leaves')
        assert dendrogram.get('color_list')

    def test_quasi_diagnalization(self):
        """
        Test the quasi-diagnalisation step of HRP algorithm.
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, asset_names=self.data.columns)
        assert hrp.ordered_indices == [13, 9, 10, 8, 14, 7, 1, 6, 4, 16, 3, 17,
                                       12, 18, 22, 0, 15, 21, 11, 2, 20, 5, 19]

    def test_value_error_for_non_dataframe_input(self):
        """
        Test ValueError on passing non-dataframe input.
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            hrp.allocate(asset_prices=self.data.values, asset_names=self.data.columns)

    def test_value_error_for_non_date_index(self):
        """
        Test ValueError on passing dataframe not indexed by date.
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            data = self.data.reset_index()
            hrp.allocate(asset_prices=data, asset_names=self.data.columns)

    def test_resampling_asset_prices(self):
        """
        Test resampling of asset prices.
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data, resample_by='B', asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_all_inputs_none(self):
        """
        Test allocation when all inputs are None.
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            hrp.allocate(asset_names=self.data.columns)

    def test_hrp_with_input_as_returns(self):
        """
        Test HRP when passing asset returns dataframe as input.
        """

        hrp = HierarchicalRiskParity()
        returns = ReturnsEstimation().calculate_returns(asset_prices=self.data)
        hrp.allocate(asset_returns=returns, asset_names=self.data.columns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_hrp_with_input_as_covariance_matrix(self):
        """
        Test HRP when passing a covariance matrix as input.
        """

        hrp = HierarchicalRiskParity()
        returns = ReturnsEstimation().calculate_returns(asset_prices=self.data)
        hrp.allocate(asset_names=self.data.columns, covariance_matrix=returns.cov())
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_no_asset_names(self):
        """
        Test HRP when not supplying a list of asset names.
        """

        hrp = HierarchicalRiskParity()
        hrp.allocate(asset_prices=self.data)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_no_asset_names_with_asset_returns(self):
        """
        Test HRP when not supplying a list of asset names and when the user passes asset_returns.
        """

        hrp = HierarchicalRiskParity()
        returns = ReturnsEstimation().calculate_returns(asset_prices=self.data)
        hrp.allocate(asset_returns=returns)
        weights = hrp.weights.values[0]
        assert (weights >= 0).all()
        assert len(weights) == self.data.shape[1]
        np.testing.assert_almost_equal(np.sum(weights), 1)

    def test_valuerror_with_no_asset_names(self):
        """
        Test ValueError when not supplying a list of asset names and no other input.
        """

        with self.assertRaises(ValueError):
            hrp = HierarchicalRiskParity()
            returns = ReturnsEstimation().calculate_returns(asset_prices=self.data)
            hrp.allocate(asset_returns=returns.values)
