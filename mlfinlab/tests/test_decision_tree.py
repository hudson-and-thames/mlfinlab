"""
Tests decision tree class and models
"""

import unittest
import numpy as np
from sklearn import datasets
from mlfinlab.supervised_learning.decision_tree import ClassificationTree
from mlfinlab.util.data_manipulation import train_test_split
from mlfinlab.util.data_operation import accuracy_score


class TestDecisionTree(unittest.TestCase):
    """
    Test decision tree class and methods:
    1. Gini criterion
    """
    def test_classification_tree(self):
        """
        Test ClassificationTree class by training the model and measuring accuracy.
        :return:
        """
        print("-- Classification Tree --")

        data = datasets.load_iris()
        features = data.data
        target = data.target

        train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.4)

        clf = ClassificationTree()
        clf.fit(train_features, train_target)
        pred_target = clf.predict(test_features)

        self.assertTrue(len(pred_target) == test_features.shape[0])

        accuracy = accuracy_score(test_target, pred_target)

        print("Accuracy:", accuracy)

