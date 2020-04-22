"""
Test various functions regarding chapter 8: MDI, MDA, SFI importance.
"""
import os
import unittest

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score, log_loss
from mlfinlab.feature_importance.importance import (mean_decrease_impurity,
                                                    mean_decrease_accuracy, single_feature_importance,
                                                    plot_feature_importance)
from mlfinlab.feature_importance.orthogonal import feature_pca_analysis, get_orthogonal_features
from mlfinlab.clustering.feature_clusters import get_feature_clusters
from mlfinlab.util.generate_dataset import get_classification_data

# pylint: disable=invalid-name
class TestFeatureImportance(unittest.TestCase):
    """
    Test Feature importance
    """

    def setUp(self):
        """
        Generate X, y datasets and fit a RF
        """
        #Generate datasets
        self.X, self.y = get_classification_data(40,5,30,10000,sigmaStd=.125)
        # Fit a RF
        self.clf_base = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                               class_weight='balanced_subsample')

        self.bag_clf = BaggingClassifier(base_estimator=self.clf_base, max_features=1.0, n_estimators=100,
                                         oob_score=True, random_state=1)
        self.fit_clf = self.bag_clf.fit(self.X, self.y)
        self.cv_gen = KFold(n_splits=3, random_state=0)

    def test_orthogonal_features(self):
        """
        Test orthogonal features: PCA features, importance vs PCA importance analysis
        """

        pca_features = get_orthogonal_features(self.X)

        # PCA features should have mean of 0
        self.assertAlmostEqual(np.mean(pca_features[:, 2]), 0, delta=1e-7)
        self.assertAlmostEqual(np.mean(pca_features[:, 5]), 0, delta=1e-7)
        self.assertAlmostEqual(np.mean(pca_features[:, 6]), 0, delta=1e-7)

        # Check particular PCA values std
        self.assertAlmostEqual(np.std(pca_features[:, 1]), 1.2503, delta=0.2)
        self.assertAlmostEqual(np.std(pca_features[:, 3]), 1.0292, delta=0.2)
        self.assertAlmostEqual(np.std(pca_features[:, 4]), 1.0134, delta=0.2)

        mdi_feat_imp = mean_decrease_impurity(self.fit_clf, self.X.columns)
        pca_corr_res = feature_pca_analysis(self.X, mdi_feat_imp)

        # Check correlation metrics results
        self.assertAlmostEqual(pca_corr_res['Weighted_Kendall_Rank'][0], -0.0724, delta=1e-1)

    def test_feature_importance(self):
        """
        Test features importance: MDI, MDA, SFI and plot function
        """

        # MDI feature importance
        mdi_feat_imp = mean_decrease_impurity(self.fit_clf, self.X.columns)

        # MDA feature importance
        mda_feat_imp_log_loss = mean_decrease_accuracy(self.bag_clf, self.X, self.y, self.cv_gen,
                                                       sample_weight_train=np.ones((self.X.shape[0],)),
                                                       sample_weight_score=np.ones((self.X.shape[0],)),
                                                       scoring=log_loss)

        mda_feat_imp_f1 = mean_decrease_accuracy(self.bag_clf, self.X, self.y,
                                                 self.cv_gen, scoring=f1_score)
        # SFI feature importance
        sfi_feat_imp_log_loss = single_feature_importance(self.bag_clf, self.X,
                                                          self.y, cv_gen=self.cv_gen,
                                                          sample_weight_train=np.ones((self.X.shape[0],)),
                                                          scoring=log_loss)
        sfi_feat_imp_f1 = single_feature_importance(self.bag_clf, self.X,
                                                    self.y, cv_gen=self.cv_gen,
                                                    sample_weight_score=np.ones((self.X.shape[0],)),
                                                    scoring=f1_score)

        #Clustered feature importance
        #Auto number of clusters selection
        clustered_subsets_linear = get_feature_clusters(self.X, dependence_metric='linear',
                                                        distance_metric='angular', linkage_method='single',
                                                        n_clusters=None)

        clustered_mdi = mean_decrease_impurity(self.fit_clf, self.X.columns,
                                               clustered_subsets=clustered_subsets_linear)
        clustered_mda = mean_decrease_accuracy(self.bag_clf, self.X, self.y, self.cv_gen,
                                               clustered_subsets=clustered_subsets_linear)
        #CFI with number clusters equal to number of features
        #This is done verify the theory that if number clusters is equal to number of features then the
        #result will be same as MDA
        feature_subset_single = [[x] for x in self.X.columns]
        mda_cfi_single = mean_decrease_accuracy(self.bag_clf, self.X, self.y, self.cv_gen,
                                                clustered_subsets=feature_subset_single)

        # MDI assertions
        self.assertAlmostEqual(mdi_feat_imp['mean'].sum(), 1, delta=0.001)
        # The most informative features
        self.assertAlmostEqual(mdi_feat_imp.loc['I_1', 'mean'], 0.47075, delta=0.01)
        self.assertAlmostEqual(mdi_feat_imp.loc['I_0', 'mean'], 0.09291, delta=0.01)
        # Redundant feature
        self.assertAlmostEqual(mdi_feat_imp.loc['R_0', 'mean'], 0.07436, delta=0.01)
        # Noisy feature
        self.assertAlmostEqual(mdi_feat_imp.loc['N_0', 'mean'], 0.01798, delta=0.01)

        # MDA(log_loss) assertions
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['I_1', 'mean'], 0.59684, delta=0.1)
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['R_0', 'mean'], 0.13177, delta=0.1)

        # MDA(f1) assertions
        self.assertAlmostEqual(mda_feat_imp_f1.loc['I_1', 'mean'], 0.52268, delta=0.1)
        self.assertAlmostEqual(mda_feat_imp_f1.loc['I_2', 'mean'], 0.29533, delta=0.1)

        # SFI(log_loss) assertions
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['I_0', 'mean'], -6.50385, delta=0.1)
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['R_0', 'mean'], -3.27282, delta=0.1)

        # SFI(accuracy) assertions
        self.assertAlmostEqual(sfi_feat_imp_f1.loc['I_0', 'mean'], 0.48530, delta=0.1)
        self.assertAlmostEqual(sfi_feat_imp_f1.loc['I_1', 'mean'], 0.78778, delta=0.1)

        #MDI CFI assertions
        self.assertAlmostEqual(mdi_cfi_linear.loc['I_1', 'mean'], 0.5, delta=3)
        self.assertAlmostEqual(mdi_cfi_linear.loc['I_0', 'mean'], 0.1, delta=3)

        #MDA CFI(log_loss) assertions
        self.assertAlmostEqual(mda_cfi_linear.loc['I_1', 'mean'], 0.2, delta=3)
        self.assertAlmostEqual(mda_cfi_linear.loc['R_0', 'mean'], 0.3, delta=3)

        #Test if CFI with number of clusters same to number features is equal to MDA results
        self.assertEqual(mda_feat_imp_log_loss.loc['I_1', 'mean'], mda_cfi_single.loc['I_1', 'mean'])
        self.assertEqual(mda_feat_imp_log_loss.loc['R_0', 'mean'], mda_cfi_single.loc['R_0', 'mean'])

    def test_plot_feature_importance(self):
        """
        Test plot_feature_importance function
        """
        oos_score = cross_val_score(self.bag_clf, self.X, self.y, cv=self.cv_gen, scoring='accuracy').mean()

        mdi_feat_imp = mean_decrease_impurity(self.bag_clf, self.X.columns)
        plot_feature_importance(mdi_feat_imp, oob_score=self.bag_clf.oob_score_, oos_score=oos_score)
        plot_feature_importance(mdi_feat_imp, oob_score=self.bag_clf.oob_score_, oos_score=oos_score,
                                save_fig=True, output_path='test.png')
        os.remove('test.png')
