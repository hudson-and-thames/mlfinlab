import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class ModelFingerprint:
    def __init__(self, clf: object, X: pd.DataFrame, y: pd.Series, num_values=None):
        self.X = X
        self.y = y
        self.clf = clf

        if num_values is None:
            self.num_values = self.X.shape[0]
        else:
            self.num_values = num_values

        self.features = X.columns

        # Get possible feature values (Step 1)
        self.feature_values = {}  # Dictionary of feature values range used to build dependence functions
        for feature in self.features:
            values = np.random.choice(self.X[feature], size=self.num_values)
            self.feature_values[feature] = values

        # DataFrame with partial dependence function values for columns from X (f_k)
        self.ind_partial_dep_functions = pd.DataFrame(index=list(range(self.num_values)), columns=self.X.columns)

        self.feature_column_position_mapping = dict(zip(self.X.columns, range(0, self.X.shape[1])))

        # DataFrame with partial dependence function values for columns pairwise interactions from X (f_k_l)
        feature_pairs = []
        for pair in itertools.combinations(self.features, 2):
            feature_pairs.append('{}_{}'.format(pair[0], pair[1]))
        self.pairwise_partial_dep_functions = pd.DataFrame(index=list(range(self.num_values ** 2)),
                                                           columns=feature_pairs)

        # Get partial dependency functions
        self._get_individual_partial_dependence()
        self.pairwise_x_values = self._get_pairwise_partial_dependence()

    def _get_individual_partial_dependence(self):
        """
        Get individual partial dependence function values for each column.
        :return: None
        """
        for col in self.features:
            y_mean_arr = []  # Array of mean(prediction) for each feature value
            for x_k in self.feature_values[col]:
                # Step 2
                col_k_position = self.feature_column_position_mapping[col]
                X_ = self.X.values.copy()
                X_[:, col_k_position] = x_k

                # Step 3
                y_pred = self.clf.predict(X_)
                y_pred_mean = y_pred.mean()

                y_mean_arr.append(y_pred_mean)

            self.ind_partial_dep_functions[col] = y_mean_arr

    def _get_pairwise_partial_dependence(self):
        """
        Get pairwise partial dependence function values for feature pairs.
        :return: None
        """
        x_values_dict = {}  # Dict of pair1: [(x_k, x_l), ...], pair2: [...]
        for pair in self.pairwise_partial_dep_functions.columns:
            col_k = pair.split('_')[0]
            col_l = pair.split('_')[1]

            x_values = []  # Array of (x_k, x_l values)
            y_mean_arr = []  # Array of mean(prediction) for each feature value
            for x_k in self.feature_values[col_k]:
                for x_l in self.feature_values[col_l]:
                    col_k_position = self.feature_column_position_mapping[col_k]
                    col_l_position = self.feature_column_position_mapping[col_l]
                    X_ = self.X.values.copy()
                    X_[:, col_k_position] = x_k
                    X_[:, col_l_position] = x_l

                    x_values.append((x_k, x_l))

                    y_pred = self.clf.predict(X_)
                    y_pred_mean = y_pred.mean()
                    y_mean_arr.append(y_pred_mean)

            self.pairwise_partial_dep_functions[pair] = y_mean_arr
            x_values_dict[pair] = x_values

        return x_values_dict

    def _get_linear_effect_estimation(self):
        store = {}
        for col in self.features:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            y_mean = np.mean(y)
            linear_effect = np.mean(np.abs(lmodel.predict(x).T[0] - y_mean))
            store[col] = np.array([linear_effect])
        return store

    def _get_non_linear_effect_estimation(self):
        store = {}
        for col in self.features:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            nonlinear_effect = np.mean(np.abs(lmodel.predict(x).T[0] - y.values))
            store[col] = np.array([nonlinear_effect])
        return store

    def fit(self):
        linear_effect = self._get_linear_effect_estimation()
        non_linear_effect = self._get_non_linear_effect_estimation()
