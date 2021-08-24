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
        pass

    def fit(self, model: object, X: pd.DataFrame, num_values: int = 50, pairwise_combinations: list = None) -> None:
        """
        Get linear, non-linear and pairwise effects estimation.

        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.
        """

        pass

    def get_effects(self) -> Tuple:
        """
        Return computed linear, non-linear and pairwise effects. The model should be fit() before using this method.

        :return: (tuple) Linear, non-linear and pairwise effects, of type dictionary (raw values and normalised).
        """

        pass

    def plot_effects(self) -> plt.figure:
        """
        Plot each effect (normalized) on a bar plot (linear, non-linear). Also plots pairwise effects if calculated.

        :return: (plt.figure) Plot figure.
        """

        pass

    def _get_feature_values(self, X: pd.DataFrame, num_values: int) -> None:
        """
        Step 1 of the algorithm which generates possible feature values used in analysis.

        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        """

        pass

    def _get_individual_partial_dependence(self, model: object, X: pd.DataFrame) -> None:
        """
        Get individual partial dependence function values for each column.

        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        """

        pass

    def _get_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get linear effect estimates as the mean absolute deviation of the linear predictions around their average value.

        :param X: (pd.DataFrame) Dataframe of features.
        :return: (dict) Linear effect estimates for each feature column.
        """

        pass

    def _get_non_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get non-linear effect estimates as as the mean absolute deviation of the total marginal (single variable)
        effect around its corresponding linear effect.

        :param X: (pd.DataFrame) Dataframe of features.
        :return: (dict) Non-linear effect estimates for each feature column.
        """

        pass

    def _get_pairwise_effect(self, pairwise_combinations: list, model: object, X: pd.DataFrame, num_values) -> dict:
        """
        Get pairwise effect estimates as the de-meaned joint partial prediction of the two variables minus the de-meaned
        partial predictions of each variable independently.

        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.
        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        :return: (dict) Raw and normalised pairwise effects.
        """

        pass

    @abstractmethod
    def _get_model_predictions(self, model: object, X_: pd.DataFrame):
        """
        Get model predictions based on problem type (predict for regression, predict_proba for classification).

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """

        pass

    @staticmethod
    def _normalize(effect: dict) -> dict:
        """
        Normalize effect values (sum equals 1).

        :param effect: (dict) Effect values.
        :return: (dict) Normalized effect values.
        """

        pass


class RegressionModelFingerprint(AbstractModelFingerprint):
    """
    Regression Fingerprint class used for regression type of models.
    """

    def __init__(self):
        """
        Regression model fingerprint constructor.
        """

        pass

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """

        pass


class ClassificationModelFingerprint(AbstractModelFingerprint):
    """
    Classification Fingerprint class used for classification type of models.
    """

    def __init__(self):
        """
        Classification model fingerprint constructor.
        """

        pass

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """

        pass
