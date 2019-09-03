"""
Module which implements feature importance algorithms described in Chapter 8
"""

from mlfinlab.feature_importance.importance import (feature_importance_mean_decrease_accuracy,
                                                    feature_importance_mean_decrease_impurity, feature_importance_sfi,
                                                    plot_feature_importance)
from mlfinlab.feature_importance.orthogonal import feature_pca_analysis, get_pca_rank_weighted_kendall_tau, \
                                                   get_orthogonal_features
