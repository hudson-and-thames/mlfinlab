"""
Implementation of an algorithm described in Yimou Li, David Turkington, Alireza Yazdani
'Beyond the Black Box: An Intuitive Approach to Investment Prediction with Machine Learning'
(https://jfds.pm-research.com/content/early/2019/12/11/jfds.2019.1.023)
"""

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# pylint: disable=invalid-name
# pylint: disable=too-many-locals

class AbstractModelFingerprint(ABC):
    """
    Model fingerprint constructor.

    This is an abstract base class for the RegressionModelFingerprint and ClassificationModelFingerprint classes.
    """

    def __init__(self):
        """
        Model fingerprint constructor.
        """

        # Effects containers
        self.linear_effect = None
        self.non_linear_effect = None
        self.pair_wise_effect = None

        self.ind_partial_dep_functions = None  # partial dependence function values containers
        self.feature_column_position_mapping = None  # numerical mapping between columns string and its position
        self.feature_values = None  # Feature values used in fingerprints analysis

    def fit(self, model: object, X: pd.DataFrame, num_values: int = 50, pairwise_combinations: list = None) -> None:
        """
        Get linear, non-linear and pairwise effects estimation.

        :param model: (object) trained model.
        :param X: (pd.DataFrame) of features.
        :param num_values: (int) number of values used to estimate feature effect.
        :param pairwise_combinations: (list) of tuples (feature_i, feature_j) to test pairwise effect.
        """

        # Step 1
        self._get_feature_values(X, num_values)

        # Get partial dependency functions
        self._get_individual_partial_dependence(model, X)

        linear_effect = self._get_linear_effect(X)
        non_linear_effect = self._get_non_linear_effect(X)

        if pairwise_combinations is not None:
            pairwise_effect = self._get_pairwise_effect(pairwise_combinations, model, X, num_values)
            self.pair_wise_effect = {'raw': pairwise_effect, 'norm': self._normalize(pairwise_effect)}

        # Save results
        self.linear_effect = {'raw': linear_effect, 'norm': self._normalize(linear_effect)}
        self.non_linear_effect = {'raw': non_linear_effect, 'norm': self._normalize(non_linear_effect)}

    def get_effects(self) -> Tuple:
        """
        Return computed linear, non-linear and pairwise effects. The model should be fit() before using this method.

        :return: (tuple) of linear, non-linear and pairwise effects, of type dictionary (raw values and normalised).
        """
        return self.linear_effect, self.non_linear_effect, self.pair_wise_effect

    def plot_effects(self) -> plt.figure:
        """
        Plot each effect (normalized) on a bar plot (linear, non-linear). Also plots pairwise effects if calculated.

        :return: (plt.figure) plot figure.
        """
        if self.pair_wise_effect is None:
            fig, (ax1, ax2) = plt.subplots(2, 1)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax3.set_title('Pair-wise effect')
            ax3.bar(*zip(*self.pair_wise_effect['norm'].items()))

        ax1.set_title('Linear effect')
        ax1.bar(*zip(*self.linear_effect['norm'].items()))

        ax2.set_title('Non-Linear effect')
        ax2.bar(*zip(*self.non_linear_effect['norm'].items()))

        fig.tight_layout()
        return fig

    def _get_feature_values(self, X: pd.DataFrame, num_values: int) -> None:
        """
        Step 1 of the algorithm which generates possible feature values used in analysis.

        :param X: (pd.DataFrame) of features.
        :param num_values: (int) number of values used to estimate feature effect.
        """

        # Get possible feature values (Step 1)
        self.feature_values = {}  # Dictionary of feature values range used to build dependence functions
        for feature in X.columns:
            values = []
            for q in np.linspace(0, 1, num_values):
                values.append(np.quantile(X[feature], q=q))
            self.feature_values[feature] = np.array(values)

        # DataFrame with partial dependence function values for columns from X (f_k)
        self.ind_partial_dep_functions = pd.DataFrame(index=list(range(num_values)), columns=X.columns)

        self.feature_column_position_mapping = dict(zip(X.columns, range(0, X.shape[1])))

    def _get_individual_partial_dependence(self, model: object, X: pd.DataFrame) -> None:
        """
        Get individual partial dependence function values for each column.

        :param model: (object) trained model.
        :param X: (pd.DataFrame) of features.
        """
        for col in X.columns:
            y_mean_arr = []  # Array of mean(prediction) for each feature value
            for x_k in self.feature_values[col]:
                # Step 2
                col_k_position = self.feature_column_position_mapping[col]
                X_ = X.values.copy()
                X_[:, col_k_position] = x_k

                # Step 3
                y_pred = self._get_model_predictions(model, X_)
                y_pred_mean = np.mean(y_pred)

                y_mean_arr.append(y_pred_mean)

            self.ind_partial_dep_functions[col] = y_mean_arr

    def _get_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get linear effect estimates as the mean absolute deviation of the linear predictions around their average value.

        :param X: (pd.DataFrame) of features.
        :return: (dict) of linear effect estimates for each feature column.
        """
        store = {}
        for col in X.columns:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            y_mean = np.mean(y)
            linear_effect = np.mean(np.abs(lmodel.predict(x) - y_mean))
            store[col] = linear_effect
        return store

    def _get_non_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get non-linear effect estimates as as the mean absolute deviation of the total marginal (single variable)
        effect around its corresponding linear effect.

        :param X: (pd.DataFrame) of features.
        :return: (dict) of non-linear effect estimates for each feature column.
        """
        store = {}
        for col in X.columns:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            nonlinear_effect = np.mean(np.abs(lmodel.predict(x) - y.values))
            store[col] = nonlinear_effect
        return store

    def _get_pairwise_effect(self, pairwise_combinations: list, model: object, X: pd.DataFrame, num_values) -> dict:
        """
        Get pairwise effect estimates as the de-meaned joint partial prediction of the two variables minus the de-meaned
        partial predictions of each variable independently.

        :param pairwise_combinations: (list) of tuples (feature_i, feature_j) to test pairwise effect.
        :param model: (object) trained model.
        :param X: (pd.DataFrame) of features.
        :param num_values: (int) number of values used to estimate feature effect.
        :return: (dict) of raw and normalised pairwise effects.
        """
        store = {}

        for pair in pairwise_combinations:
            function_values = []  # Array of pairwise interactions [f_k_l(not centered), f_k(centered), f_l(centered)]
            col_k = pair[0]
            col_l = pair[1]

            y_cdf_k_centered = self.ind_partial_dep_functions[col_k] - np.mean(self.ind_partial_dep_functions[col_k])
            y_cdf_l_centered = self.ind_partial_dep_functions[col_l] - np.mean(self.ind_partial_dep_functions[col_l])

            for x_k, y_cdf_k in zip(self.feature_values[col_k], y_cdf_k_centered):
                for x_l, y_cdf_l in zip(self.feature_values[col_l], y_cdf_l_centered):
                    col_k_position = self.feature_column_position_mapping[col_k]
                    col_l_position = self.feature_column_position_mapping[col_l]
                    X_ = X.values.copy()
                    X_[:, col_k_position] = x_k
                    X_[:, col_l_position] = x_l

                    y_cdf_k_l = self._get_model_predictions(model, X_).mean()

                    function_values.append([y_cdf_k_l, y_cdf_k, y_cdf_l])

            function_values = np.array(function_values)  # Convert to np.array to vectorize operations

            # Cumulative pairwise effect value for a given feature
            # Need to center f_k_l as f_k, f_l have been already centered
            centered_y_cdf_k_l = function_values[:, 0] - np.mean(function_values[:, 0])

            # See the paper to make use of notation
            f_k = function_values[:, 1]
            f_l = function_values[:, 2]
            func_value = sum(abs((centered_y_cdf_k_l - f_k - f_l)))

            store[str(pair)] = func_value / (num_values ** 2)

        return store

    @abstractmethod
    def _get_model_predictions(self, model: object, X_: pd.DataFrame):
        """
        Get model predictions based on problem type (predict for regression, predict_proba for classification).

        :param model: (object) trained model.
        :param X_: (np.array) feature set.
        :return: (np.array) of predictions.
        """
        raise NotImplementedError('Must implement _get_model_predictions')

    @staticmethod
    def _normalize(effect: dict) -> dict:
        """
        Normalize effect values (sum equals 1).

        :param effect: (dict) of effect values.
        :return: (dict) of normalized effect values.
        """
        values_sum = sum(effect.values())
        updated_effect = {}

        for k, v in effect.items():
            updated_effect[k] = v / values_sum
        return updated_effect


class RegressionModelFingerprint(AbstractModelFingerprint):
    """
    Regression Fingerprint class used for regression type of models.
    """

    def __init__(self):
        """
        Regression model fingerprint constructor.
        """
        AbstractModelFingerprint.__init__(self)

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) trained model.
        :param X_: (np.array) feature set.
        :return: (np.array) of predictions.
        """
        return model.predict(X_)


class ClassificationModelFingerprint(AbstractModelFingerprint):
    """
    Classification Fingerprint class used for classification type of models.
    """

    def __init__(self):
        """
        Classification model fingerprint constructor.
        """
        AbstractModelFingerprint.__init__(self)

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) trained model.
        :param X_: (np.array) feature set.
        :return: (np.array) of predictions.
        """
        return model.predict_proba(X_)[:, 1]
