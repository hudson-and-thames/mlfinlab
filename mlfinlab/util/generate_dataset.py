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
    A funtion to generate synthetic classification datasets

    :param n_features: (int) Total number of features to be generated (i.e. informative + redundant + noisy).
    :param n_informative: (int) Number of informative features.
    :param n_redundant: (int) Number of redundant features.
    :param n_samples: (int) Number of samples (rows) to be generate.
    :param random_state: (int) Random seed.
    :param sigma: (float) use this argument to introduce substitution effect  to the redundant features in
                     the dataset by adding gaussian noise. The lower the  value of  sigma, the  greater the
                     substitution effect.
    :return: (pd.DataFrame, pd.Series)  X and y as features and labels repectively.
    """
    np.random.seed(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_features-n_redundant, n_informative=n_informative,
                               n_redundant=0, shuffle=False, random_state=random_state)
    cols = ['I_'+str(i) for i in range(n_informative)]
    cols += ['N_'+str(i) for i in range(n_features-n_informative-n_redundant)]
    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)
    i = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X['R_'+str(k)] = X['I_'+str(j)]+np.random.normal(size=X.shape[0])*sigma
    return X, y
