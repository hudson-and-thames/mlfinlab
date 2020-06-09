"""
Calculate Stationarity.
"""

from statsmodels.tsa.stattools import adfuller


def calc_stationarity(data, maxlag=None, regression="c", autolag="AIC", store=False, regresults=False):
    """
    Wrapper function for Augmented Dickey-Fuller unit root test, directly forked from
    `statsmodels.tsa.stattools.adfuller
    <https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html>`_.

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

    :param data: (pd.Series) The data series to test.

    :param maxlag: (int) Maximum lag which is included in test, default 12*(nobs/100)^{1/4}.

    :param regression: (str) ("c","ct","ctt","nc") Constant and trend order to include in
        regression.

        - "c" : constant only (default).
        - "ct" : constant and trend.
        - "ctt" : constant, and linear and quadratic trend.
        - "nc" : no constant, no trend.

    :param autolag: (str) ("AIC", "BIC", "t-stat", None) Method to use when automatically
        determining the lag.

        - If None, then maxlag lags are used.
        - If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        - "If t-stat" based choice of maxlag, it starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.

    :param store: (bool) If True, then a result instance is returned additionally to the
            adf statistic. Default is False.

    :param regresults: (bool) This is optional, and if True, the full regression results are returned.
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
    return adfuller(data, maxlag=maxlag, regression=regression, autolag=autolag, store=store,
                    regresults=regresults)
