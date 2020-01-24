"""
Implements model fingerprint algorithm from 'Beyond the Black Box' paper
https://jfds.pm-research.com/content/early/2019/12/11/jfds.2019.1.023
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# pylint: disable=invalid-name

class RegressionModelFingerprint:
    """
    Regression Fingerprint class
    """

    def __init__(self, model: object, X: pd.DataFrame, num_values=50):
        """
        Constructs Regression model fingerprint.
        :param model: (object) trained regression model
        :param X: (pd.DataFrame) of features
        :param num_values: (int) number of values to fix for each feature
        """

        self.X = X
        self.model = model

        # Effects containers
        self.linear_effect = None
        self.non_linear_effect = None
        self.pair_wise_effect = None

        self.num_values = num_values

        # TODO: drop binary features?
        # Get possible feature values (Step 1)
        self.feature_values = {}  # Dictionary of feature values range used to build dependence functions
        for feature in self.X.columns:
            values = []
            for q in np.linspace(0, 1, self.num_values):
                values.append(np.quantile(self.X[feature], q=q))
            self.feature_values[feature] = np.array(values)

        # DataFrame with partial dependence function values for columns from X (f_k)
        self.ind_partial_dep_functions = pd.DataFrame(index=list(range(self.num_values)), columns=self.X.columns)

        self.feature_column_position_mapping = dict(zip(self.X.columns, range(0, self.X.shape[1])))

        # Get partial dependency functions
        self._get_individual_partial_dependence()

    def _get_individual_partial_dependence(self) -> None:
        """
        Get individual partial dependence function values for each column.
        :return: None
        """
        for col in self.X.columns:
            y_mean_arr = []  # Array of mean(prediction) for each feature value
            for x_k in self.feature_values[col]:
                # Step 2
                col_k_position = self.feature_column_position_mapping[col]
                X_ = self.X.values.copy()
                X_[:, col_k_position] = x_k

                # Step 3
                y_pred = self.model.predict(X_)

                y_pred_mean = y_pred.mean()

                y_mean_arr.append(y_pred_mean)

            self.ind_partial_dep_functions[col] = y_mean_arr

    def _get_linear_effect_estimation(self) -> dict:
        """
        Get linear effect estimates.
        :return: (dict) of linear effect estimates for each feature column.
        """
        store = {}
        for col in self.X.columns:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            y_mean = np.mean(y)
            linear_effect = np.mean(np.abs(lmodel.predict(x) - y_mean))
            store[col] = linear_effect
        return store

    def _get_non_linear_effect_estimation(self) -> dict:
        """
        Get non-linear effect estimates.
        :return: (dict) of non-linear effect estimates for each feature column
        """
        store = {}
        for col in self.X.columns:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            nonlinear_effect = np.mean(np.abs(lmodel.predict(x) - y.values))
            store[col] = nonlinear_effect
        return store

    def get_pairwise_effect(self, combinations: list) -> dict:
        """
        Get pairwise effect estimates.
        :param combinations: (list) of tuples (feature_i, feature_j) to test pairwise effect
        """
        store = {}

        for pair in combinations:
            col_k = pair[0]
            col_l = pair[1]

            func_value = 0  # Cumulative pairwise effect value for a given feature
            for x_k, y_cdf_k in zip(self.feature_values[col_k], self.ind_partial_dep_functions[col_k]):
                for x_l, y_cdf_l in zip(self.feature_values[col_l], self.ind_partial_dep_functions[col_l]):
                    col_k_position = self.feature_column_position_mapping[col_k]
                    col_l_position = self.feature_column_position_mapping[col_l]
                    X_ = self.X.values.copy()
                    X_[:, col_k_position] = x_k
                    X_[:, col_l_position] = x_l

                    y_pred = self.model.predict(X_)
                    y_cdf_k_l = y_pred.mean()

                    func_value += abs(y_cdf_k_l - y_cdf_k - y_cdf_l)

            store[str(pair)] = func_value / (self.num_values ** 2)

        self.pair_wise_effect = _normalize(store)

    def fit(self) -> None:
        """
        Get linear, non-linear effects estimation.
        :return: None
        """
        self.linear_effect = _normalize(self._get_linear_effect_estimation())
        self.non_linear_effect = _normalize(self._get_non_linear_effect_estimation())

    def plot_effects(self) -> None:
        """
        Plot each effect on a bar plot, plots only top n_features pairwise effects.
        :return: None
        """
        if self.pair_wise_effect is None:
            fig, (ax1, ax2) = plt.subplots(2, 1)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax3.set_title('Pair-wise effect')
            ax3.bar(*zip(*self.pair_wise_effect.items()))

        ax1.set_title('Linear effect')
        ax1.bar(*zip(*self.linear_effect.items()))

        ax2.set_title('Non-Linear effect')
        ax2.bar(*zip(*self.non_linear_effect.items()))

        fig.tight_layout()
        plt.show()


def _normalize(effect: dict) -> dict:
    """
    Normalize effect values (sum equals 1)
    :param effect: (dict) of effect values
    :return: (dict) of normalized effect values
    """
    values_sum = sum(effect.values())
    updated_effect = {}
    if values_sum > 1e-3:
        for k, v in effect.items():
            updated_effect[k] = v / values_sum
    else:
        updated_effect = effect
    return updated_effect
