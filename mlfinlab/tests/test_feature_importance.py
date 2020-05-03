"""
Test various functions regarding chapter 8: MDI, MDA, SFI importance.
"""
import os
import unittest

import numpy as np

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
        self.X, self.y = get_classification_data(10, 5, 2, 1000, random_state=0, sigma=0)
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
        self.assertAlmostEqual(np.std(pca_features[:, 1]), 1.3813, delta=0.2)
        self.assertAlmostEqual(np.std(pca_features[:, 3]), 1.0255, delta=0.2)
        self.assertAlmostEqual(np.std(pca_features[:, 4]), 1.0011, delta=0.2)

        mdi_feat_imp = mean_decrease_impurity(self.fit_clf, self.X.columns)
        pca_corr_res = feature_pca_analysis(self.X, mdi_feat_imp)

        # Check correlation metrics results
        self.assertAlmostEqual(pca_corr_res['Weighted_Kendall_Rank'][0], 0.7424, delta=1e-1)

    def test_feature_importance(self):
        """
        Test features importance: MDI, MDA, SFI and plot function
        """
        #getting the clustered subsets for CFI with number of clusters selection using ONC algorithm
        clustered_subsets_linear = get_feature_clusters(self.X, dependence_metric='linear',
                                                        distance_metric=None, linkage_method=None,
                                                        n_clusters=None)
        #Also to verify the theory that if number clusters is equal to number of features then the
        #result will be same as MDA
        feature_subset_single = [[x] for x in self.X.columns]

        # MDI feature importance
        mdi_feat_imp = mean_decrease_impurity(self.fit_clf, self.X.columns)
        #Clustered MDI feature importance
        clustered_mdi = mean_decrease_impurity(self.fit_clf, self.X.columns,
                                               clustered_subsets=clustered_subsets_linear)
        mdi_cfi_single = mean_decrease_impurity(self.fit_clf, self.X.columns,
                                                clustered_subsets=feature_subset_single)

        # MDA feature importance
        mda_feat_imp_log_loss = mean_decrease_accuracy(self.bag_clf, self.X, self.y, self.cv_gen,
                                                       sample_weight_train=np.ones((self.X.shape[0],)),
                                                       sample_weight_score=np.ones((self.X.shape[0],)),
                                                       scoring=log_loss)

        mda_feat_imp_f1 = mean_decrease_accuracy(self.bag_clf, self.X, self.y,
                                                 self.cv_gen, scoring=f1_score)
        #ClusteredMDA feature importance
        clustered_mda = mean_decrease_accuracy(self.bag_clf, self.X, self.y, self.cv_gen,
                                               clustered_subsets=clustered_subsets_linear)
        mda_cfi_single = mean_decrease_accuracy(self.bag_clf, self.X, self.y, self.cv_gen,
                                                clustered_subsets=feature_subset_single)

        # SFI feature importance
        sfi_feat_imp_log_loss = single_feature_importance(self.bag_clf, self.X,
                                                          self.y, cv_gen=self.cv_gen,
                                                          sample_weight_train=np.ones((self.X.shape[0],)),
                                                          scoring=log_loss)
        sfi_feat_imp_f1 = single_feature_importance(self.bag_clf, self.X,
                                                    self.y, cv_gen=self.cv_gen,
                                                    sample_weight_score=np.ones((self.X.shape[0],)),
                                                    scoring=f1_score)

        # MDI assertions
        self.assertAlmostEqual(mdi_feat_imp['mean'].sum(), 1, delta=0.001)
        # The most informative features
        self.assertAlmostEqual(mdi_feat_imp.loc['I_1', 'mean'], 0.48058, delta=0.01)
        self.assertAlmostEqual(mdi_feat_imp.loc['I_0', 'mean'], 0.08214, delta=0.01)
        # Redundant feature
        self.assertAlmostEqual(mdi_feat_imp.loc['R_0', 'mean'], 0.06511, delta=0.01)
        # Noisy feature
        self.assertAlmostEqual(mdi_feat_imp.loc['N_0', 'mean'], 0.02229, delta=0.01)

        # MDA(log_loss) assertions
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['I_1', 'mean'], 0.65522, delta=0.1)
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['R_0', 'mean'], 0.00332, delta=0.1)

        # MDA(f1) assertions
        self.assertAlmostEqual(mda_feat_imp_f1.loc['I_1', 'mean'], 0.47751, delta=0.1)
        self.assertAlmostEqual(mda_feat_imp_f1.loc['I_2', 'mean'], 0.33617, delta=0.1)

        # SFI(log_loss) assertions
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['I_0', 'mean'], -6.39442, delta=0.1)
        self.assertAlmostEqual(sfi_feat_imp_log_loss.loc['R_0', 'mean'], -5.04315, delta=0.1)

        # SFI(accuracy) assertions
        self.assertAlmostEqual(sfi_feat_imp_f1.loc['I_0', 'mean'], 0.48915, delta=0.1)
        self.assertAlmostEqual(sfi_feat_imp_f1.loc['I_1', 'mean'], 0.78443, delta=0.1)

        #Cluster MDI  assertions
        self.assertAlmostEqual(clustered_mdi.loc['R_0', 'mean'], 0.01912, delta=0.1)
        self.assertAlmostEqual(clustered_mdi.loc['I_0', 'mean'], 0.06575, delta=0.1)

        #Clustered MDA (log_loss) assertions
        self.assertAlmostEqual(clustered_mda.loc['I_0', 'mean'], 0.04154, delta=0.1)
        self.assertAlmostEqual(clustered_mda.loc['R_0', 'mean'], 0.02940, delta=0.1)

        #Test if CFI with number of clusters same to number features is equal to normal MDI & MDA results
        self.assertAlmostEqual(mdi_feat_imp.loc['I_1', 'mean'], mdi_cfi_single.loc['I_1', 'mean'], delta=0.1)
        self.assertAlmostEqual(mdi_feat_imp.loc['R_0', 'mean'], mdi_cfi_single.loc['R_0', 'mean'], delta=0.1)
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['I_1', 'mean'], mda_cfi_single.loc['I_1', 'mean'], delta=0.1)
        self.assertAlmostEqual(mda_feat_imp_log_loss.loc['R_0', 'mean'], mda_cfi_single.loc['R_0', 'mean'], delta=0.1)

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
