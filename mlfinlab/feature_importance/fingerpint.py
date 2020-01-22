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

        # Effects containers
        self.linear_effect = None
        self.non_linear_effect = None
        self.pair_wise_effect = None

        if len(self.y.unique()) == 2:
            self.model_type = 'classification'
        else:
            self.model_type = 'regression'

        if num_values is None:
            self.num_values = self.X.shape[0]
        else:
            self.num_values = num_values

        self.features = X.columns

        # TODO: take into account feature values, or drop them
        # Get possible feature values (Step 1)
        self.feature_values = {}  # Dictionary of feature values range used to build dependence functions
        for feature in self.features:
            values = []
            for q in np.linspace(0, 1, self.num_values):
                values.append(np.quantile(self.X[feature], q=q))
            self.feature_values[feature] = np.array(values)

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

    def _get_individual_partial_dependence(self) -> None:
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
                if self.model_type == 'classification':
                    y_pred = pd.DataFrame(self.clf.predict_proba(X_))[0]
                else:
                    y_pred = self.clf.predict(X_)

                y_pred_mean = y_pred.mean()

                y_mean_arr.append(y_pred_mean)

            self.ind_partial_dep_functions[col] = y_mean_arr

    def _get_linear_effect_estimation(self) -> dict:
        """
        Get linear effect estimates
        :return: (dict) of linear effect estimates for each feature column
        """
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

    def _get_non_linear_effect_estimation(self) -> dict:
        """
        Get non-linear effect estimates.
        :return: (dict) of non-linear effect estimates for each feature column
        """
        store = {}
        for col in self.features:
            x = self.feature_values[col].reshape(-1, 1)
            y = self.ind_partial_dep_functions[col]

            lmodel = LinearRegression(fit_intercept=True, normalize=False)
            lmodel.fit(x, y)
            nonlinear_effect = np.mean(np.abs(lmodel.predict(x).T[0] - y.values))
            store[col] = np.array([nonlinear_effect])
        return store

    def _get_pairwise_effect_estimation(self) -> dict:
        """
        Get pairwise effect estimates.
        :return: (dict) of pairwise effect estimates for each feature column.
        """
        store = {}
        for pair in self.pairwise_partial_dep_functions.columns:
            col_k = pair.split('_')[0]
            col_l = pair.split('_')[1]

            func_value = 0  # cumulative pairwise effect value for a given feature
            for x_k, y_cdf_k in zip(self.feature_values[col_k], self.ind_partial_dep_functions[col_k]):
                for x_l, y_cdf_l in zip(self.feature_values[col_l], self.ind_partial_dep_functions[col_l]):
                    col_k_position = self.feature_column_position_mapping[col_k]
                    col_l_position = self.feature_column_position_mapping[col_l]
                    X_ = self.X.values.copy()
                    X_[:, col_k_position] = x_k
                    X_[:, col_l_position] = x_l

                    y_pred = self.clf.predict(X_)
                    y_cdf_k_l = y_pred.mean()

                    func_value += abs(y_cdf_k_l - y_cdf_k - y_cdf_l)

            store[pair] = [func_value]

        return store

    def fit(self) -> None:
        """
        Get linear, non-linear, pairwise effects estimation.
        :return: None
        """
        self.linear_effect = self._get_linear_effect_estimation()
        self.non_linear_effect = self._get_non_linear_effect_estimation()
        self.pair_wise_effect = self._get_pairwise_effect_estimation()

    def plot_effect(self, effect_dict: dict) -> None:
        """
        Plot the effect on a bar plot.
        :param effect_dict: (dict) of effect values for each feature
        :return: None
        """

        pd.DataFrame(effect_dict).T.plot(kind='bar', title='Effect plot', legend=False)
        plt.ylabel('Effect values')
        plt.xlabel('Features')
        plt.show()
