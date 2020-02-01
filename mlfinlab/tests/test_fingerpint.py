"""
Test RegressionModelFingerprint and ClassificationModelFingerprint implementations.
"""

import unittest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston, load_breast_cancer
from mlfinlab.feature_importance import RegressionModelFingerprint, ClassificationModelFingerprint


# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

class TestModelFingerprint(unittest.TestCase):
    """
    Test model fingerprint functions
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """

        self.X, self.y = load_boston(return_X_y=True)
        self.X = pd.DataFrame(self.X[:100])
        self.y = pd.Series(self.y[:100])

        self.reg_rf = RandomForestRegressor(n_estimators=10, random_state=42)
        self.reg_linear = LinearRegression(fit_intercept=True, normalize=False)
        self.reg_rf.fit(self.X, self.y)
        self.reg_linear.fit(self.X, self.y)

        self.reg_fingerprint = RegressionModelFingerprint()

    def test_linear_effect(self):
        """
        Test get_linear_effect for various regression models and num_values.
        """

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        linear_effect, _, _ = self.reg_fingerprint.get_effects()

        # Test the most informative feature effects for reg_rf
        informative_features_1 = [0, 5, 6, 12]
        for feature, effect_value in zip(informative_features_1, [0.0577, 0.5102, 0.136, 0.2139]):
            self.assertAlmostEqual(linear_effect['norm'][feature], effect_value, delta=1e-3)

        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20)
        linear_effect, _, _ = self.reg_fingerprint.get_effects()

        # Test the most informative feature effects for reg_linear
        informative_features_2 = [0, 2, 4, 5, 6]
        for feature, effect_value in zip(informative_features_2, [0.13, 0.0477, 0.1, 0.4, 0.208]):
            self.assertAlmostEqual(linear_effect['norm'][feature], effect_value, delta=1e-3)

        # Test fingerprints with bigger num_values
        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=70)
        linear_effect_70, _, _ = self.reg_fingerprint.get_effects()

        # Increasing the number of samples doesn't change feature effect massively
        for feature in informative_features_1:
            self.assertAlmostEqual(linear_effect['norm'][feature],
                                   linear_effect_70['norm'][feature], delta=0.05)

    def test_non_linear_effect(self):
        """
        Test get_non_linear_effect for various regression models and num_values.
        """

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        _, non_linear_effect, _ = self.reg_fingerprint.get_effects()

        # Test the most informative feature effects for reg_rf
        informative_features_1 = [0, 5, 6, 12]
        for feature, effect_value in zip(informative_features_1, [0.0758, 0.3848, 0.1, 0.28]):
            self.assertAlmostEqual(non_linear_effect['norm'][feature], effect_value, delta=1e-3)

        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20)
        _, non_linear_effect, _ = self.reg_fingerprint.get_effects()

        # Non-linear effect to linear model is zero
        for effect_value in non_linear_effect['raw'].values():
            self.assertAlmostEqual(effect_value, 0, delta=1e-8)

        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=70)
        _, non_linear_effect_70, _ = self.reg_fingerprint.get_effects()

        # Increasing the number of samples doesn't change feature effect massively
        for feature in informative_features_1:
            self.assertAlmostEqual(non_linear_effect['raw'][feature],
                                   non_linear_effect_70['raw'][feature], delta=0.05)

    def test_pairwise_effect(self):
        """
        Test compute_pairwise_effect for various regression models and num_values.
        """

        combinations = [(0, 5), (0, 12), (1, 2), (5, 7), (3, 6), (4, 9)]
        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20, pairwise_combinations=combinations)
        _, _, pair_wise_effect = self.reg_fingerprint.get_effects()

        for pair, effect_value in zip(combinations, [0.203, 0.17327, 0.005, 0.032, 0, 0.00004]):
            self.assertAlmostEqual(pair_wise_effect['raw'][str(pair)], effect_value, delta=1e-3)

        combinations = [(0, 5), (0, 12), (1, 2), (5, 7), (3, 6), (4, 9)]
        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20, pairwise_combinations=combinations)
        _, _, pair_wise_effect = self.reg_fingerprint.get_effects()

        # Pairwise effect for linear model should be zero
        for pair in combinations:
            self.assertAlmostEqual(pair_wise_effect['raw'][str(pair)], 0, delta=1e-9)

    def test_classification_fingerpint(self):
        """
        Test model fingerprint values (linear, non-linear, pairwise) for classification model.
        """

        X, y = load_breast_cancer(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        clf_fingerpint = ClassificationModelFingerprint()
        clf_fingerpint.fit(clf, X, num_values=20, pairwise_combinations=[(0, 1), (2, 3), (8, 9)])

        linear_effect, non_linear_effect, pair_wise_effect = clf_fingerpint.get_effects()

        for feature, effect in zip([0, 2, 3, 8, 9], [0.0068, 0.0249, 0.014, 0]):
            self.assertAlmostEqual(linear_effect['raw'][feature], effect, delta=1e-3)

        for feature, effect in zip([0, 2, 3, 8, 9], [0.0062, 0.0217, 0.0155, 0.0013]):
            self.assertAlmostEqual(non_linear_effect['raw'][feature], effect, delta=1e-3)

        for comb, effect in zip([(0, 1), (2, 3), (8, 9)], [0.008, 0.0087, 0]):
            self.assertAlmostEqual(pair_wise_effect['raw'][str(comb)], effect, delta=1e-3)

    def test_plot_effects(self):
        """
        Test plot_effects function.
        """

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        self.reg_fingerprint.plot_effects()

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20, pairwise_combinations=[(1, 2), (3, 5)])
        self.reg_fingerprint.plot_effects()
