'''
This module generates  synthetic classification dataset of INFORMED, REDUNDANT, and NOISE explanatory
variables based on the book Machine Learning for Asset Manager (code snippet 6.1)
'''
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# pylint: disable=invalid-name
def get_classification_data(n_features=100, n_informative=25, n_redundant=25, n_samples=10000, random_state=0, sigma=.0):
    """
    A function to generate synthetic classification datasets

    :param n_features: (int) Total number of features to be generated (i.e. informative + redundant + noisy).
    :param n_informative: (int) Number of informative features.
    :param n_redundant: (int) Number of redundant features.
    :param n_samples: (int) Number of samples (rows) to be generate.
    :param random_state: (int) Random seed.
    :param sigma: (float) This argument is used to introduce substitution effect to the redundant features in
                     the dataset by adding gaussian noise. The lower the  value of  sigma, the  greater the
                     substitution effect.
    :return: (pd.DataFrame, pd.Series)  X and y as features and labels respectively.
    """

    pass
