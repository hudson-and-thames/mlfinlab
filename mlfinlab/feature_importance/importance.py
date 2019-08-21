import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
import matplotlib as mpl


def feature_importance_mean_imp_reduction(clf, feature_names):
    # Feature importance based on IS mean impurity reduction
    feature_imp_df = {i: tree.feature_importances_ for i, tree in enumerate(clf.estimators_)}
    feature_imp_df = pd.DataFrame.from_dict(feature_imp_df, orient='index')
    feature_imp_df.columns = feature_names
    feature_imp_df = feature_imp_df.replace(0, np.nan)  # Because max_features = 1
    imp = pd.concat({'mean': feature_imp_df.mean(), 'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -.5},
                    axis=1)
    imp /= imp['mean'].sum()
    return imp


def feature_importance_mean_decrease_accuracy(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    # Feature importance based on OOS score reduction
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')

    cross_validator = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # Purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)

    for i, (train, test) in enumerate(cross_validator.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        pred = fit.predict(X1)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values,
                                    labels=clf.classes_)
        elif scoring == 'accuracy_score':
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values,
                                           labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)

    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    else:
        imp = imp / (1. - scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    return imp, scr0.mean()


def feature_importance_sfi(featNames, clf, trnsX, cont, scoring, cvGen):
    # TODO: parallel
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'],
                      scoring=scoring, cvGen=cvGen)
    imp.loc[featName, 'mean'] = df0.mean()
    imp.loc[featName, 'std'] = df0.std() * df0.shape[0] ** -.5
    return imp


def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    mpl.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'],
                          error_kw={'ecolor': 'r'})
    if method == 'MDI':
        mpl.xlim([0, imp.sum(axis=1).max()])
        mpl.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index): ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center',
                                                    va='center',
                                                    color='black')
    mpl.title('tag = ' + tag + ' | simNum = ' + str(simNum) + ' | oob = ' + str(round(oob, 4)) +
              ' | oos = ' + str(round(oos, 4)))
    mpl.savefig(pathOut + 'featImportance_' + str(simNum) + '.png', dpi=100)
    mpl.clf()
    mpl.close()
    return
