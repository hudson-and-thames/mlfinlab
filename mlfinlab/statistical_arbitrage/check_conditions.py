"""
Calculate Stationarity and Cointegration.
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint


def calc_stationarity(x, maxlag=None, regression="c", autolag="AIC", store=False, regresults=False):
    """
    Wrapper function for Augmented Dickey-Fuller unit root test, directly forked from
    statsmodels.tsa.stattools.adfuller.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    :param x: (pd.Sereis) The data series to test.
    :param maxlag: (int) Maximum lag which is included in test, default 12*(nobs/100)^{1/4}.
    :param regression: (str) {"c","ct","ctt","nc"} Constant and trend order to include in
        regression.
        - "c" : constant only (default).
        - "ct" : constant and trend.
        - "ctt" : constant, and linear and quadratic trend.
        - "nc" : no constant, no trend.
    :param autolag: (str) {"AIC", "BIC", "t-stat", None} Method to use when automatically
        determining the lag.
        - If None, then maxlag lags are used.
        - If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        - "If t-stat" based choice of maxlag, it starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
    :param store: (bool) If True, then a result instance is returned additionally to the
        adf statistic. Default is False.
    :param regresults: (bool) This is optional, and ff True, the full regression results are returned.
        Default is False.
    :return: (tuple) ADF, P-value, Usedlag, Nobs, Critical Values, Icbest, Resstore
        - adf: (float) The test statistic.
        - pvalue: (float) MacKinnon"s approximate p-value based on MacKinnon (1994, 2010).
        - usedlag: (int) The number of lags used.
        - nobs: (int) The number of observations used for the ADF regression and calculation of
        the critical values.
        - critical values: (dict) Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
        - icbest: (float) The maximized information criterion if autolag is not None.
        - resstore: (ResultStore) This is optional. A dummy class with results attached as attributes.
    """
    return adfuller(x, maxlag=maxlag, regression=regression, autolag=autolag, store=store,
                    regresults=regresults)


def calc_cointegration(y0, y1, trend="c", method="aeg", maxlag=None, autolag="aic",
                       return_results=None):
    """
    Wrapper function for augmented Engle-Granger two-step cointegration test, directly forked from
    statsmodels.tsa.stattools.coint.

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

    :param y0: (pd.Series) The first element in cointegrated system. Must be 1-d.
    :param y1: (pd.Series) The remaining elements in cointegrated system.
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
    return coint(y0, y1, trend=trend, method=method, maxlag=maxlag, autolag=autolag,
                 return_results=return_results)
