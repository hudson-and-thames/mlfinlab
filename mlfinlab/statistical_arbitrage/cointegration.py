"""
Calculate Cointegration.
"""

from statsmodels.tsa.stattools import coint


def calc_cointegration(data1, data2, trend="c", method="aeg", maxlag=None, autolag="aic",
                       return_results=None):
    """
    Wrapper function for augmented Engle-Granger two-step cointegration test, directly forked
    from `statsmodels.tsa.stattools.coint
    <https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html>`.

    Test for no-cointegration of a univariate equation.

    The null hypothesis is no cointegration. Variables in y0 and y1 are
    assumed to be integrated of order 1, I(1).

    This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in 1st stage regression, i.e. in
    cointegrating equation.

    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values and critical values are obtained through regression surface
    approximation from MacKinnon 1994 and 2010.

    If the two series are almost perfectly collinear, then computing the
    test is numerically unstable. However, the two series will be cointegrated
    under the maintained assumption that they are integrated. In this case
    the t-statistic will be set to -inf and the pvalue to zero.

    :param data1: (pd.Series) The first element in cointegrated system. Must be 1-d.
    :param data2: (pd.Series) The remaining elements in cointegrated system.
    :param trend: (str) {"c","ct","ctt","nc"} Constant and trend order to include in
        regression.
        - "c" : constant only (default).
        - "ct" : constant and trend.
        - "ctt" : constant, and linear and quadratic trend.
        - "nc" : no constant, no trend.
    :param method: (str) {"aeg"} Only "aeg" (augmented Engle-Granger) is available.
    :param maxlag: None or int Argument for `adfuller`, largest or given number of lags.
    :param autolag: (str) Argument for `adfuller`, lag selection criterion.
        - If None, then maxlag lags are used without lag search.
        - If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        - "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
    :param return_results: (bool) For future compatibility, currently only tuple available.
        If True, then a results instance is returned. Otherwise, a tuple
        with the test outcome is returned. Set `return_results=False` to
        avoid future changes in return.

    :return: (tuple) Coint_t, Pvalue, Crit_value
    - coint_t: (float) The t-statistic of unit-root test on residuals.
    - pvalue: (float) MacKinnon"s approximate, asymptotic p-value based on MacKinnon (1994).
    - crit_value: (dict) Critical values for the test statistic at the 1 %, 5 %, and 10 %
    levels based on regression curve. This depends on the number of observations.
    """
    return coint(data1, data2, trend=trend, method=method, maxlag=maxlag, autolag=autolag,
                 return_results=return_results)
