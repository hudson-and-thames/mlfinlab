# pylint: disable=anomalous-backslash-in-string
"""
Calculate Stationarity.
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


def calc_adfuller(data, maxlag=None, regression="c", autolag="AIC", store=False, regresults=False):
    """
    Wrapper function for Augmented Dickey-Fuller unit root test, directly forked from
    `statsmodels.tsa.stattools.adfuller
    <https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html>`_.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    .. math::
        H_0: \delta = 1

        H_1: \delta < 1


    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the p-value is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    :param data: (pd.Series) The data series to test.

    :param maxlag: (int) Maximum lag which is included in test, default :math:`\\frac{12 * nobs}{
        100}^{\\frac{1}{4}}`.

    :param regression: (str) (``c``, ``ct``, ``ctt``, ``nc``) Constant/Trend order to include in regression.

        - ``c`` : constant only (default).
        - ``ct`` : constant and trend.
        - ``ctt`` : constant, and linear and quadratic trend.
        - ``nc`` : no constant, no trend.

    :param autolag: (str) (``AIC``, ``BIC``, ``t-stat``, ``None``) Method to use when automatically
        determining the lag.

        - If ``None``, then maxlag lags are used.
        - If ``AIC`` (default) or ``BIC``, then the number of lags is chosen to minimize the
          corresponding information criterion.
        - If ``t-stat``, it starts with maxlag and drops a lag until the t-statistic on the
          last lag length is significant using a 5%-sized test.

    :param store: (bool) If ``True``, then a result instance is returned additionally to the
        adf statistic. Default is ``False``.

    :param regresults: (bool) This is optional, and if ``True``, the full regression results are returned.
        Default is ``False``.

    :return: (tuple) ``ADF``, ``P-value``, ``Usedlag``, ``Nobs``, ``Critical Values``, ``Icbest``, ``Resstore``.

        - ``adf``: (float) The test statistic.
        - ``pvalue``: (float) MacKinnon"s approximate p-value based on MacKinnon
          (`1994 <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.894.8429&rep=rep1&type=pdf>`_,
          `2010 <https://www.econstor.eu/bitstream/10419/67744/1/616664753.pdf>`_).
        - ``usedlag``: (int) The number of lags used.
        - ``nobs``: (int) The number of observations used for the ADF regression and calculation of
          the critical values.
        - ``critical values``: (dict) Critical values for the test statistic at the 1 %, 5 %, and 10 %
          levels. Based on MacKinnon (2010).
        - ``icbest``: (float) The maximized information criterion if autolag is not None.
        - ``resstore``: (ResultStore) This is optional. A dummy class with results attached as attributes.

    """
    return adfuller(data, maxlag=maxlag, regression=regression, autolag=autolag, store=store,
                    regresults=regresults)


def calc_kpss(data, regression='c', nlags=None, store=False):
    """
    Wrapper function for Kwiatkowski-Phillips-Schmidt-Shin unit root test, directly forked from
    `statsmodels.tsa.stattools.kpss
    <https://www.statsmodels.org/stable/_modules/statsmodels/tsa/stattools.html#kpss>`_.

    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null
    hypothesis that x is level or trend stationary.

    :param data: (pd.Series) The data series to test.

    :param regression: (str) {``c``, ``ct``} The null hypothesis for the KPSS test.

        - ``c`` : (Default) The data is stationary around a constant.
        - ``ct`` : The data is stationary around a trend.

    :param nlags: (str) {``None``, ``str``, ``int``}, (optional) Indicates the number of lags to be
        used.

        - If ``None`` (default), lags is calculated using the legacy method.
        - If ``auto``, lags is calculated using the data-dependent method of Hobijn et al.
        - If ``legacy``,  uses :math:`\\frac{12*n}{100}^{.25}` , as outlined in
          `Schwert (1989) <https://ideas.repec.org/p/nbr/nberte/0073.html>`_.

    :param store: (bool) If ``True``, then a result instance is returned additionally to
        the KPSS statistic (default is False).

    :return: (tuple) ``kpss_stat``, ``p-value``, ``lags``, ``crit``, ``resstore``.

        - ``kpss_stat``: (float) The KPSS test statistic.
        - ``p_value``: (float) The p-value of the test. The p-value is interpolated from Table 1 in
          Kwiatkowski et al. (1992), and a boundary point is returned if the test statistic is
          outside the table of critical values, that is, if the p-value is outside the
          interval (0.01, 0.1).
        - ``lags``: (int) The truncation lag parameter.
        - ``crit``: (dict) The critical values at 10%, 5%, 2.5% and 1%. Based on Kwiatkowski et al.
          (1992).
        - ``resstore``: (optional) instance of ResultStore. An instance of a dummy class with results
          attached as attributes.

    .. note::
        To estimate :math:`\sigma^2` the Newey-West estimator is used. If lags is None,
        the truncation lag parameter is set to :math:`\\frac{12 * n}{100}^{\\frac{1}{4}}`,
        as outlined in Schwert (1989). The p-values are interpolated from
        Table 1 of `Kwiatkowski et al. (1992) <http://debis.deu.edu.tr/userweb/onder.hanedar/dosyalar/kpss.pdf>`_.
        If the computed statistic is outside the table of critical values, then a warning message is
        generated.

    Missing values are not handled.
    """
    return kpss(data, regression=regression, nlags=nlags, store=store)
