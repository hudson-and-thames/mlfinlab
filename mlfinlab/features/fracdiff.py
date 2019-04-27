
import numpy as np
import pandas as pd


# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
# https://github.com/SimonOuellette35/FractionalDiff/blob/master/question2.py

def get_weights(d, size):
    """
    Source: Chapter 5, AFML (section 5.4.2)
    The helper function generates weights that are used to compute fractionally differentiated series.
    It computes the weights that get used in the computation of  fractionally
    differentiated series.  This generates a non-terminating series that
    approaches zero asymptotically.  The side effect of this function is that
    it leads to negative drift "caused by an expanding window's added weights"
    (see page 83 AFML)

    When d is real (non-integer) positive number then it preserves memory.

    The book does not discuss what should be expected if d is a negative real
    number.  Conceptually (from set theory) negative d leads to set of negative
    number of elements.  And that translates into a set whose elements can be
    selected more than once or as many times as one chooses (multisets with
    unbounded multiplicity) - see http://faculty.uml.edu/jpropp/msri-up12.pdf.

    :param d: (float) differencing amount
    :param size: (int) length of the series
    :return: (ndarray) weight vector
    """
    # thresh > 0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)

    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff(series, d, thresh=0.01):
    """
    Source: Chapter 5, AFML (section 5.5);

    References:
    https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
    https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
    https://en.wikipedia.org/wiki/Fractional_calculus

    The steps are as follows:
    - Compute weights (this is a one-time exercise)
    - Iteratively apply the weights to the price series and generate output points

    This is the expanding window variant of the fracDiff algorithm
    Note 1: For thresh-1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarility bounded [0, 1]

    :param series: (pd.Series) a time series that needs to be differenced
    :param d: (float) Differencing amount
    :param thresh: (float) threshold or epsilon
    :return: (pd.DataFrame) data frame of differenced series
    """

    # 1. Compute weights for the longest series
    w = get_weights(d, series.shape[0])

    # 2. Determine initial calculations to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thresh].shape[0]

    # 3. Apply weights to values
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series(index=series.index)

        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs
            else:
                df_[loc] = np.dot(w[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def get_weights_ffd(d, thresh, lim):
    """
    Source: Chapter 5, AFML (section 5.4.2)
    The helper function generates weights that are used to compute fractionally differentiated series.
    It computes the weights that get used in the computation of fractionally
    differentiated series. The series is fixed width and same wrights (generated
    by this function) can be used when creating fractional differentiated series.
    This makes the process more efficient.  But the side-effect is that the
    fractionally differentiated series is skewed and has excess kurtosis ...
    in other words, it is not Gaussian any more.

    The discussion of positive and negative d is similar to that in get_weights
    (see the function get_weights)

    :param d: (float) differencing amount
    :param thresh: (float) threshold for minimum weight
    :param lim: (int) maximum length of the weight vector
    :return: (ndarray) weight vector
    """

    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_ffd(series, d, thresh=1e-5):
    """
    Source: Chapter 5, AFML (section 5.5);
    References:
    https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
    https://wwwf.imperial.ac.uk/~ejm/M3S8/Problems/hosking81.pdf
    https://en.wikipedia.org/wiki/Fractional_calculus

    The steps are as follows:
    - Compute weights (this is a one-time exercise)
    - Iteratively apply the weights to the price series and generate output points

    Constant width window (new solution)
    Note 1: thresh determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarity bounded [0, 1].

    :param series: (pd.Series)
    :param d: (float) differencing amount
    :param thresh: (float) threshold for minimum weight
    :return: (pd.DataFrame) a data frame of differenced series
    """

    # 1) Compute weights for the longest series
    w = get_weights_ffd(d, thresh, series.shape[0])
    width = len(w) - 1

    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series(index=series.index)
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            else:
                df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

