"""
Tests decision tree class and models
"""

import unittest
import os
import numpy as np
import pandas as pd

from mlfinlab.supervised_learning.decision_tree import ClassificationTree
from mlfinlab.util.data_manipulation import train_test_split, standardize
from mlfinlab.util.data_operation import mean_squared_error, calculate_variance, accuracy_score
from mlfinlab.util.misc import Plot

from sklearn import datasets


class TestDecisionTree(unittest.TestCase):
    """
    Test decision tree class and methods:
    1. Gini criterion
    """
    def __init__(self):
        self.setUp()


    def setUp(self):
        """
        Set the file path for the tick data csv
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + '/test_data/tick_data.csv'

    def load_data(self):
        df = pd.read_csv(self.path)
        self.data = df

    def test_classification_tree(self):
        print("-- Classification Tree --")

        data = datasets.load_iris()
        X = data.data
        y = data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        clf = ClassificationTree()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)

        Plot().plot_in_2d(X_test, y_pred,
                          title="Decision Tree",
                          accuracy=accuracy,
                          legend_labels=data.target_names)

if __name__ == '__main__':
    np.random.seed(111)
    testTree = TestDecisionTree()
    testTree.test_classification_tree()
